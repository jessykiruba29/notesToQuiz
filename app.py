import streamlit as st
import fitz  # PyMuPDF
import random
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# ------------------ Configuration ------------------
st.set_page_config(
    page_title="Notes → Quiz Generator",
    page_icon="📘",
    layout="wide"
)

# ------------------ Model Loader (cached) ------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_t5_model():
    try:
        model_name = "google/flan-t5-small"
        local_dir = "./models/flan-t5-small"
        
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(local_dir, "config.json")):
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            # Save locally for future use
            tokenizer.save_pretrained(local_dir)
            model.save_pretrained(local_dir)
            
        return tokenizer, model
    except Exception as e:
        st.error(f"⚠️ Could not load T5 model: {e}")
        return None, None

# ------------------ Optimized Helpers ------------------

def clean_text(text):
    """Efficient text cleaning."""
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\?\!]', '', text)
    return text.strip()

def extract_text_from_pdf(uploaded_file):
    """Efficient PDF text extraction with progress tracking."""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text_chunks = []
        total_pages = len(doc)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num, page in enumerate(doc):
            status_text.text(f"📄 Processing page {page_num + 1} of {total_pages}...")
            text = page.get_text("text").strip()
            if text and len(text) > 10:  # Only add non-empty pages
                text_chunks.append(text)
            progress_bar.progress((page_num + 1) / total_pages)
            time.sleep(0.01)  # Small delay to show progress
        
        status_text.empty()
        progress_bar.empty()
        doc.close()
        
        return " ".join(text_chunks)
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        return ""

def extract_questions_from_text(text):
    """
    Improved rule-based extraction of Q&A pairs with better patterns.
    """
    questions = []
    
    # Split into sentences more reliably
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 4:  # Skip very short sentences
            continue
        
        # Pattern 1: "X is/are Y" patterns
        patterns = [
            (r"^(.*?\b\w+)\s+is\s+(.*?\.?)$", "What is {}?"),
            (r"^(.*?\b\w+)\s+are\s+(.*?\.?)$", "What are {}?"),
            (r"^(.*?\b\w+)\s+was\s+(.*?\.?)$", "What was {}?"),
            (r"^(.*?\b\w+)\s+were\s+(.*?\.?)$", "What were {}?"),
            (r"^(.*?\b\w+)\s+means?\s+(.*?\.?)$", "What does {} mean?"),
            (r"^(.*?\b\w+)\s+refers to\s+(.*?\.?)$", "What does {} refer to?"),
        ]
        
        for pattern, question_template in patterns:
            match = re.match(pattern, sent, re.IGNORECASE)
            if match:
                subject, description = match.groups()
                # Clean up the subject and description
                subject = subject.strip().rstrip(',')
                description = description.strip().rstrip('.')
                
                # Avoid questions that are too long or too short
                if 3 <= len(subject.split()) <= 15 and len(description.split()) >= 2:
                    question = question_template.format(subject)
                    questions.append({
                        "question": question,
                        "answer": description,
                        "source_sentence": sent
                    })
                    break  # Only use one pattern per sentence
    
    return questions

def generate_smart_distractors(correct_answer, num_distractors=3):
    """Generate more realistic distractors."""
    words = correct_answer.split()
    distractors = set()
    
    if len(words) <= 1:
        # Fallback for very short answers
        base_words = ["different", "alternative", "various", "other", "similar"]
        for _ in range(num_distractors):
            fake = f"{random.choice(base_words)} {correct_answer}"
            distractors.add(fake)
    else:
        # Strategy 1: Shuffle words
        attempts = 0
        while len(distractors) < num_distractors and attempts < 50:
            shuffled = words.copy()
            random.shuffle(shuffled)
            fake = " ".join(shuffled)
            if fake != correct_answer and 0.5 <= len(fake.split()) / len(words) <= 2:
                distractors.add(fake)
            attempts += 1
        
        # Strategy 2: Replace with similar length random words
        if len(distractors) < num_distractors:
            similar_length_words = ["concept", "idea", "theory", "method", "process", 
                                  "system", "approach", "technique", "principle"]
            for _ in range(num_distractors - len(distractors)):
                fake = " ".join(random.sample(similar_length_words, min(3, len(words))))
                distractors.add(fake)
    
    return list(distractors)[:num_distractors]

def make_mcqs(text, max_questions=20):
    """Turn text into MCQs with limits for performance."""
    with st.spinner("🔍 Extracting questions from text..."):
        qa_pairs = extract_questions_from_text(text)
    
    if not qa_pairs:
        st.warning("⚠️ No questions could be automatically extracted from the PDF. Try a textbook or content-rich document.")
        return []
    
    # Limit number of questions for performance
    qa_pairs = qa_pairs[:max_questions]
    
    mcqs = []
    progress_bar = st.progress(0)
    
    for i, qa in enumerate(qa_pairs):
        distractors = generate_smart_distractors(qa["answer"])
        options = distractors + [qa["answer"]]
        random.shuffle(options)
        
        mcqs.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "options": options,
            "source": qa.get("source_sentence", "")
        })
        
        progress_bar.progress((i + 1) / len(qa_pairs))
    
    progress_bar.empty()
    return mcqs

def display_question(q, q_index, progress, total):
    """Render one question with options."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### 📝 Question {q_index + 1} of {total}")
        st.markdown(f"**{q['question']}**")
    
    with col2:
        st.caption(f"Progress: {progress}/{total}")
    
    # Initialize session state for this question
    sel_key = f"selected_option_{q_index}"
    submit_key = f"submitted_{q_index}"
    
    if sel_key not in st.session_state:
        st.session_state[sel_key] = None
    if submit_key not in st.session_state:
        st.session_state[submit_key] = False
    
    # Display options
    option_labels = [f"**{chr(65+i)}.** {option}" for i, option in enumerate(q['options'])]
    
    selected_index = st.radio(
        "Choose your answer:",
        options=option_labels,
        index=None if not st.session_state[submit_key] else q['options'].index(st.session_state[sel_key]) if st.session_state[sel_key] in q['options'] else None,
        key=f"radio_{q_index}",
        disabled=st.session_state[submit_key]
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    result = None
    
    with col1:
        if st.button("Submit Answer", key=f"submit_{q_index}", disabled=st.session_state[submit_key] or selected_index is None):
            if selected_index is None:
                st.warning("Please select an answer first.")
            else:
                selected_option = q['options'][option_labels.index(selected_index)]
                st.session_state[sel_key] = selected_option
                st.session_state[submit_key] = True
                
                if selected_option == q["answer"]:
                    st.success("✅ Correct!")
                    result = True
                else:
                    st.error(f"❌ Incorrect. The correct answer is: **{q['answer']}**")
                    result = False
    
    with col2:
        if st.session_state[submit_key] and q_index < total - 1:
            if st.button("Next Question →", key=f"next_{q_index}"):
                st.session_state.current_q += 1
                st.rerun()
    
    with col3:
        if st.button("Show Source", key=f"source_{q_index}"):
            st.info(f"**Source sentence:** {q.get('source', 'Not available')}")
    
    return result

# ------------------ Enhanced Flashcards & Analytics ------------------

def show_flashcards(questions):
    st.subheader("📇 Flashcards")
    
    if not questions:
        st.info("No questions available. Upload a PDF first.")
        return
    
    # Add flip animation with CSS
    st.markdown("""
    <style>
    .flashcard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    for i, q in enumerate(questions):
        with st.expander(f"Card {i+1}: {q['question'][:50]}...", expanded=False):
            st.markdown(f'<div class="flashcard"><strong>Q:</strong> {q["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="flashcard"><strong>A:</strong> {q["answer"]}</div>', unsafe_allow_html=True)
            
            if q.get('source'):
                st.caption(f"Source: {q['source'][:100]}...")

def show_analytics(questions):
    st.subheader("📊 Analytics")
    
    if not questions:
        st.info("No quiz data available. Complete the quiz first.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Questions", len(questions))
    
    with col2:
        # Calculate score based on session state
        correct_answers = 0
        for i in range(len(questions)):
            if st.session_state.get(f"submitted_{i}", False):
                selected = st.session_state.get(f"selected_option_{i}")
                if selected == questions[i]["answer"]:
                    correct_answers += 1
        
        if len(questions) > 0:
            score = (correct_answers / len(questions)) * 100
            st.metric("Your Score", f"{score:.1f}%")
        else:
            st.metric("Your Score", "0%")
    
    with col3:
        attempted = sum(1 for i in range(len(questions)) if st.session_state.get(f"submitted_{i}", False))
        st.metric("Questions Attempted", f"{attempted}/{len(questions)}")
    
    # Progress chart
    if attempted > 0:
        st.progress(attempted / len(questions))

# ------------------ Main Application ------------------

def main():
    st.title("📘 Notes → Quiz Generator")
    st.markdown("Transform your study materials into interactive quizzes!")
    
    # Initialize session state
    if "current_q" not in st.session_state:
        st.session_state.current_q = 0
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        if st.button("🔄 Reset Quiz", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("selected_option_") or key.startswith("submitted_"):
                    del st.session_state[key]
            st.session_state.current_q = 0
            st.rerun()
        
        if st.button("📊 Show All Answers", use_container_width=True):
            st.session_state.show_answers = True
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app converts your PDF notes into multiple-choice quizzes using AI.")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("🔄 Processing your PDF..."):
            # Extract text
            raw_text = extract_text_from_pdf(uploaded_file)
            
            if raw_text and len(raw_text) > 100:  # Ensure we have sufficient text
                cleaned_text = clean_text(raw_text)
                
                # Generate questions
                st.session_state.questions = make_mcqs(cleaned_text)
                st.session_state.pdf_processed = True
                
                if st.session_state.questions:
                    st.success(f"✅ Generated {len(st.session_state.questions)} questions from your PDF!")
                else:
                    st.warning("⚠️ Could not generate questions from this PDF. Try a different document.")
            else:
                st.error("❌ The PDF appears to be empty or too short. Try a different file.")
    
    # Main tabs
    if st.session_state.questions:
        tabs = st.tabs(["📖 Quiz", "📇 Flashcards", "📊 Analytics"])
        
        with tabs[0]:
            qlist = st.session_state.questions
            current_q = st.session_state.current_q
            
            if current_q < len(qlist):
                q = qlist[current_q]
                display_question(q, current_q, current_q + 1, len(qlist))
            else:
                st.success("🎉 Quiz completed! Check your results in the Analytics tab.")
                if st.button("Restart Quiz"):
                    st.session_state.current_q = 0
                    st.rerun()
        
        with tabs[1]:
            show_flashcards(st.session_state.questions)
        
        with tabs[2]:
            show_analytics(st.session_state.questions)
    else:
        if uploaded_file and st.session_state.pdf_processed:
            st.info("📝 No quiz questions could be generated from this PDF. Try a textbook or content-rich document.")
        else:
            st.info("📂 Upload a PDF to generate your quiz. Supported: textbooks, articles, study notes.")

# ------------------ Run Application ------------------
if __name__ == "__main__":
    main()
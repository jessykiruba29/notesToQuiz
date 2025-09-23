


# ...existing code...
import streamlit as st
import fitz  # PyMuPDF
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
import torch
import random
import pyttsx3
import tempfile
import os
# --- Multilingual Setup ---
def speak(text):
    """Play text as speech using pyttsx3 (offline, cross-platform)."""
    engine = pyttsx3.init()
    # Use a temporary file to save the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_path = fp.name
    engine.save_to_file(text, temp_path)
    engine.runAndWait()
    audio_file = open(temp_path, 'rb')
    audio_bytes = audio_file.read()
    audio_file.close()
    os.remove(temp_path)
    st.audio(audio_bytes, format='audio/mp3')

# --- Multilingual Setup ---
LANGUAGES = {
    'English': None,
    'Hindi': 'Helsinki-NLP/opus-mt-en-hi',
    'Tamil': 'Helsinki-NLP/opus-mt-en-ta',
    'Bengali': 'Helsinki-NLP/opus-mt-en-bn',
}

import functools
@functools.lru_cache(maxsize=8)
def get_translation_pipeline(target_lang):
    if target_lang == 'English':
        return None
    model_name = LANGUAGES[target_lang]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return lambda text: tokenizer.batch_decode(model.generate(tokenizer([text], return_tensors="pt", padding=True)["input_ids"]), skip_special_tokens=True)[0]

def translate_question(q, target_lang, translation_fn=None):
    if not translation_fn:
        return q
    q_trans = dict(q)
    q_trans['question'] = translation_fn(q['question'])
    q_trans['options'] = [translation_fn(opt) for opt in q['options']]
    q_trans['answer'] = translation_fn(q['answer'])
    return q_trans

# --- Helper Functions ---
def load_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# --- Smarter Distractor Generation (Streamlit Cloud compatible) ---
def generate_distractors(context, correct_answer, num_distractors=3):
    """Simple distractor generator using string shuffling (offline, no extra models needed)."""
    words = correct_answer.split()
    distractors = set()
    attempts = 0
    while len(distractors) < num_distractors and attempts < 20:
        random.shuffle(words)
        fake_ans = " ".join(words)
        if fake_ans != correct_answer and fake_ans.strip():
            distractors.add(fake_ans)
        attempts += 1
    # Fill remaining with random common words if needed
    while len(distractors) < num_distractors:
        distractors.add(f"{correct_answer} {random.choice(['X','Y','Z'])}")
    return list(distractors)[:num_distractors]


# --- T5 Question Generation (Streamlit Cloud compatible) ---


# --- Local or Fallback T5 Model Loading ---
import os
T5_LOCAL_PATH = "./t5-small"
FALLBACK_MODEL = "google/flan-t5-small"


# Returns (tokenizer, model, used_fallback: bool)

# --- Safe T5 Model Loading ---
@st.cache_resource(show_spinner=False)
def get_t5_model():
    """Load T5 model safely, using local if available, else fallback. Returns (tokenizer, model, used_fallback)."""
    try:
        if os.path.exists(T5_LOCAL_PATH):
            tokenizer = T5Tokenizer.from_pretrained(T5_LOCAL_PATH)
            model = T5ForConditionalGeneration.from_pretrained(T5_LOCAL_PATH)
            return tokenizer, model, False
        else:
            tokenizer = T5Tokenizer.from_pretrained(FALLBACK_MODEL)
            model = T5ForConditionalGeneration.from_pretrained(FALLBACK_MODEL)
            return tokenizer, model, True
    except Exception as e:
        st.error(f"⚠️ Could not load T5 model: {e}")
        return None, None, False  # fallback failed




def safe_generate_question(text):
    """Generate a question from text safely. Returns warning if model is not loaded."""
    tokenizer, model, _ = get_t5_model()
    if tokenizer is None or model is None:
        return "⚠️ Question generation unavailable"
    try:
        input_text = f"generate question: {text}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"⚠️ Error generating question: {e}"

def generate_questions(text, num_questions=10):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    questions = []
    for i in range(min(num_questions, len(sentences))):
        context = sentences[i]
        out = safe_generate_question(context)
        correct = out
        distractors = generate_distractors(context, correct)
        options = [correct] + distractors
        random.shuffle(options)
        questions.append({
            'question': out,
            'options': options,
            'answer': correct,
            'context': context,
            'wrong_count': 0
        })
    return questions





def display_question(q, idx, lang='English', progress=None, total=None, feedback=None):
    # q: question dict, idx: index in quiz, lang: language, progress/total: for progress bar, feedback: (None, True, False)
    if 'selected_option' not in st.session_state:
        st.session_state['selected_option'] = None
    if 'answer_submitted' not in st.session_state:
        st.session_state['answer_submitted'] = False
    if 'show_next' not in st.session_state:
        st.session_state['show_next'] = False

    st.markdown(f"<div style='margin-bottom:0.5em; font-size:1.3em; font-weight:bold;'>📝 Q{idx+1}: {q['question']}</div>", unsafe_allow_html=True)
    if progress is not None and total is not None:
        st.markdown(f"<div style='margin-bottom:0.5em; font-size:1.1em;'>Progress: <b>Question {progress} of {total}</b></div>", unsafe_allow_html=True)
        st.progress(progress / total)

    sel_key = f'selected_option_{idx}'
    sub_key = f'answer_submitted_{idx}'
    next_key = f'show_next_{idx}'
    if sel_key not in st.session_state:
        st.session_state[sel_key] = None
    if sub_key not in st.session_state:
        st.session_state[sub_key] = False
    if next_key not in st.session_state:
        st.session_state[next_key] = False

    col1, col2 = st.columns([8,1])
    with col2:
        if st.button('🔊 Play Question', key=f'play_{idx}'):
            speak(q['question'])

    st.session_state[sel_key] = st.radio(
        "Select your answer:",
        q['options'],
        index=0 if st.session_state.get(sel_key) is None else q['options'].index(st.session_state[sel_key]),
        key=f"radio_{idx}"
    )

    submit = st.button("Submit Answer", key=f"submit_{idx}")
    nextq = st.button("Next Question", key=f"next_{idx}")

    if submit and not st.session_state[sub_key]:
        st.session_state[sub_key] = True
        st.session_state[next_key] = True
        if st.session_state[sel_key] == q['answer']:
            st.success("✅ Correct!")
            return True
        else:
            st.error(f"❌ Wrong! Correct answer: {q['answer']}")
            return False
    elif st.session_state[sub_key]:
        if st.session_state[sel_key] == q['answer']:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Wrong! Correct answer: {q['answer']}")
    if nextq and st.session_state[next_key]:
        st.session_state[sub_key] = False
        st.session_state[next_key] = False
        st.session_state[sel_key] = None
        return 'next'
    return None
# --- MAIN APP ---
def main():
    st.set_page_config(page_title="AI Notes-to-Quiz Generator", layout="wide")

    st.title("AI Notes-to-Quiz Generator 📝🤖")
    st.markdown("Generate quizzes from your notes or PDFs. Adaptive, multilingual, and Streamlit Cloud compatible!")

    # Model warning in sidebar
    _, _, used_fallback = get_t5_model()
    with st.sidebar:
        st.header("Settings")
        if used_fallback:
            st.warning("⚠️ Using fallback T5 model from HuggingFace Hub. For best performance, upload the T5 model folder as ./t5-small.")
        lang = st.selectbox("Quiz Language", list(LANGUAGES.keys()), index=0)

    # Tabs
    tabs = st.tabs(["Quiz", "Flashcards", "Analytics"])

    # --- Quiz Tab ---
    with tabs[0]:
        st.subheader("Quiz")
        uploaded_file = st.file_uploader("Upload PDF Notes", type=["pdf"])
        if uploaded_file:
            text = load_text_from_pdf(uploaded_file)
            if 'questions' not in st.session_state:
                st.session_state['questions'] = generate_questions(text, num_questions=10)
                st.session_state['current_q'] = 0
                st.session_state['answers'] = []
            questions = st.session_state['questions']
            current_q = st.session_state['current_q']
            q = questions[current_q]
            # Translation
            translation_fn = get_translation_pipeline(lang)
            if lang != 'English' and translation_fn:
                q = translate_question(q, lang, translation_fn)
            feedback = None
            result = display_question(q, current_q, lang, progress=current_q+1, total=len(questions))
            if result is True or result is False:
                st.session_state['answers'].append((q, st.session_state['selected_option'], result))
                if not result:
                    questions[current_q]['wrong_count'] += 1
            if result == 'next' and current_q < len(questions)-1:
                st.session_state['current_q'] += 1
                st.experimental_rerun()
            elif result == 'next' and current_q == len(questions)-1:
                st.success("Quiz complete! View your flashcards and analytics.")
        else:
            st.info("Upload a PDF to start the quiz.")

    # --- Flashcards Tab ---
    with tabs[1]:
        show_flashcards()

    # --- Analytics Tab ---
    with tabs[2]:
        show_analytics()


# --- Analytics Function ---
def show_analytics():
    st.markdown("<h2 style='margin-bottom:1em;'>📊 Analytics</h2>", unsafe_allow_html=True)
    answers = st.session_state.get('answers', [])
    if not answers:
        st.info("No quiz attempts yet.")
        return
    total = len(answers)
    correct = sum(1 for a in answers if a[2])
    wrong = total - correct
    st.write(f"Total Questions Answered: {total}")
    st.write(f"Correct: {correct}")
    st.write(f"Wrong: {wrong}")
    st.progress(correct/total)


# --- Run App ---
if __name__ == "__main__":
    main()


def save_bookmark(question):
    if 'bookmarks' not in st.session_state:
        st.session_state['bookmarks'] = []
    if question not in st.session_state['bookmarks']:
        st.session_state['bookmarks'].append(question)
        st.success("Bookmarked!")
    else:
        st.info("Already bookmarked.")




def show_flashcards():
    st.markdown("<h2 style='margin-bottom:1em;'>⭐ Flashcards</h2>", unsafe_allow_html=True)
    bookmarks = st.session_state.get('bookmarks', [])
    answers = st.session_state.get('answers', [])
    if not bookmarks:
        st.info("No bookmarks yet.")
        return
    import hashlib
    for i, q in enumerate(bookmarks):
        q_hash = hashlib.md5(q['question'].encode()).hexdigest()[:8]
        show_key = f'show_answer_{i}_{q_hash}'
        remove_key = f'remove_bm_{i}_{q_hash}'
        if show_key not in st.session_state:
            st.session_state[show_key] = False
        with st.container():
            st.markdown(f"<div style='background:#f8f9fa; border-radius:10px; padding:1em; margin-bottom:1em; box-shadow:0 2px 8px #eee;'>", unsafe_allow_html=True)
            st.markdown(f"<b>📝 Q{i+1}:</b> {q['question']}", unsafe_allow_html=True)
            col1, col2 = st.columns([3,1])
            with col1:
                st.session_state[show_key] = st.checkbox(f"Show Answer {i+1}", value=st.session_state[show_key], key=f"cb_{show_key}")
                if st.session_state[show_key]:
                    st.markdown(f"<b>Answer:</b> <span style='color:#2980b9; font-size:1.1em'>{q['answer']}</span>", unsafe_allow_html=True)
                    user_attempts = [a for a in answers if a[0]['question'] == q['question']]
                    if user_attempts:
                        last_attempt = user_attempts[-1]
                        if last_attempt[2]:
                            st.markdown("<span style='color:#2ecc40; font-weight:bold;'>✅ Correct in quiz</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span style='color:#e74c3c; font-weight:bold;'>❌ Incorrect in quiz</span>", unsafe_allow_html=True)
            with col2:
                if st.button(f"Remove", key=remove_key):
                    st.session_state['bookmarks'].remove(q)
                    st.session_state[show_key] = False
                    st.experimental_rerun()


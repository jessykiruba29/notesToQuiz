


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


# --- Smarter Distractor Generation ---
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
@st.cache_resource(show_spinner=False)
def get_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

def safe_generate_question(text):
    tokenizer, model = get_t5_model()
    input_text = f"generate question: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    with st.container():
        st.markdown(f"<div style='margin-bottom:0.5em; font-size:1.3em; font-weight:bold;'>📝 Q{idx+1}: {q['question']}</div>", unsafe_allow_html=True)
        if progress is not None and total is not None:
            st.markdown(f"<div style='margin-bottom:0.5em; font-size:1.1em;'>Progress: <b>Question {progress} of {total}</b></div>", unsafe_allow_html=True)
            st.progress(progress / total)
        col1, col2, col3 = st.columns([5,1,1])
        with col2:
            if st.button("🔊", key=f"tts_{idx}"):
                speak(q['question'])
        with col3:
            if st.button("⭐", key=f"bm_icon_{idx}"):
                save_bookmark(q)
        st.markdown("<div style='margin-top:0.5em'></div>", unsafe_allow_html=True)
        user_answer = st.radio("<b>Choose one:</b>", q['options'], key=f"q_{idx}", format_func=lambda x: x, label_visibility="visible")
        if feedback == 'correct':
            st.markdown("<div style='color: #2ecc40; font-size:1.1em; font-weight:bold; margin-top:0.5em;'>✅ Correct! 🎉</div>", unsafe_allow_html=True)
        elif feedback == 'wrong':
            st.markdown(f"<div style='color: #e74c3c; font-size:1.1em; font-weight:bold; margin-top:0.5em;'>❌ Wrong!<br>Correct: <b>{q['answer']}</b></div>", unsafe_allow_html=True)
        return user_answer

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
    for i, q in enumerate(bookmarks):
        with st.container():
            st.markdown(f"<div style='background:#f8f9fa; border-radius:10px; padding:1em; margin-bottom:1em; box-shadow:0 2px 8px #eee;'>", unsafe_allow_html=True)
            st.markdown(f"<b>📝 Q{i+1}:</b> {q['question']}", unsafe_allow_html=True)
            col1, col2 = st.columns([3,1])
            with col1:
                if st.button(f"Show Answer {i+1}", key=f"show_ans_{i}"):
                    st.markdown(f"<b>Answer:</b> <span style='color:#2980b9; font-size:1.1em'>{q['answer']}</span>", unsafe_allow_html=True)
                    user_attempts = [a for a in answers if a[0]['question'] == q['question']]
                    if user_attempts:
                        last_attempt = user_attempts[-1]
                        if last_attempt[2]:
                            st.markdown("<span style='color:#2ecc40; font-weight:bold;'>✅ You answered this correctly in the quiz.</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span style='color:#e74c3c; font-weight:bold;'>❌ You answered this incorrectly in the quiz.</span>", unsafe_allow_html=True)
            with col2:
                if st.button(f"Remove", key=f"remove_bm_{i}"):
                    st.session_state['bookmarks'].remove(q)
                    st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

def show_analytics():
    st.subheader("Quiz Analytics")
    answers = st.session_state.get('answers', [])
    bookmarks = st.session_state.get('bookmarks', [])
    repeat_queue = st.session_state.get('repeat_queue', [])
    total = len(answers)
    correct = sum(1 for a in answers if a[2])
    percent = (correct / total * 100) if total else 0
    st.metric("Final Score", f"{correct} / {total}")
    st.metric("Percentage", f"{percent:.1f}%")
    st.metric("Bookmarked Questions", len(bookmarks))
    # Most-wrong questions
    wrongs = [a for a in answers if not a[2]]
    if wrongs:
        st.markdown("**Most Wrong Questions:**")
        for i, (q, user_ans, _) in enumerate(wrongs):
            st.markdown(f"- {q['question']} (Your answer: {user_ans}, Correct: {q['answer']})")
    else:
        st.info("No wrong answers!")


def main():
    st.set_page_config(page_title="recallai", layout="centered")
    st.title("RecallAI: Notes-to-Quiz Generator 🧠")

    # Sidebar language selector
    st.sidebar.header("🌐 Language")
    lang = st.sidebar.selectbox("Select language", list(LANGUAGES.keys()), index=0)
    translation_fn = get_translation_pipeline(lang)

    # Session state init
    for key, default in [
        ('score', 0),
        ('current_q', 0),
        ('questions', []),
        ('answers', []),
        ('bookmarks', []),
        ('repeat_queue', []),
        ('quiz_finished', False)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    tabs = st.tabs(["Quiz", "Flashcards", "Analytics"])

    # --- Quiz Tab ---
    with tabs[0]:
        uploaded = st.file_uploader("Upload PDF notes", type=["pdf"])
        if uploaded and st.button("Generate Quiz"):
            text = load_text_from_pdf(uploaded)
            st.session_state['questions'] = generate_questions(text, num_questions=10)
            st.session_state['current_q'] = 0
            st.session_state['score'] = 0
            st.session_state['answers'] = []
            st.session_state['repeat_queue'] = []
            st.session_state['quiz_finished'] = False


        questions = st.session_state.get('questions', [])
        idx = st.session_state.get('current_q', 0)
        repeat_queue = st.session_state.get('repeat_queue', [])
        quiz_finished = st.session_state.get('quiz_finished', False)


        # Adaptive repeat: sort repeat_queue by wrong_count descending
        if repeat_queue:
            repeat_queue.sort(key=lambda q: -q.get('wrong_count', 0))

        # Determine which question to show: repeat queue has priority
        total_questions = len(questions)
        answered = idx
        if questions and not quiz_finished:
            if repeat_queue:
                q = repeat_queue[0]
                is_repeat = True
                progress = answered
            elif idx < total_questions:
                q = questions[idx]
                is_repeat = False
                progress = idx + 1
            else:
                st.markdown("<div style='color:#2980b9; font-size:1.2em; font-weight:bold; margin-top:1em;'>🎉 Quiz complete! Your score: {}/{} </div>".format(st.session_state['score'], len(questions)), unsafe_allow_html=True)
                st.session_state['quiz_finished'] = True
                st.stop()

            # Translate question/options if needed
            q_disp = translate_question(q, lang, translation_fn) if lang != 'English' else q
            # Feedback logic
            feedback = None
            if 'last_feedback' in st.session_state:
                feedback = st.session_state.pop('last_feedback')
            user_answer = display_question(q_disp, idx if not is_repeat else f"repeat_{idx}", lang=lang, progress=progress, total=total_questions, feedback=feedback)
            if st.button("Submit Answer", key=f"submit_{idx}"):
                correct = user_answer == q_disp['answer']
                st.session_state['answers'].append((q, user_answer, correct))
                if correct:
                    st.session_state['score'] += 1
                    if is_repeat:
                        st.session_state['repeat_queue'].pop(0)
                    st.session_state['last_feedback'] = 'correct'
                else:
                    q['wrong_count'] = q.get('wrong_count', 0) + 1
                    if not is_repeat and q not in st.session_state['repeat_queue']:
                        st.session_state['repeat_queue'].append(q)
                    st.session_state['last_feedback'] = 'wrong'
                st.experimental_rerun()
            if st.button("Next Question", key=f"next_{idx}"):
                if not is_repeat:
                    st.session_state['current_q'] += 1
                elif is_repeat:
                    pass
                st.experimental_rerun()

        elif questions and quiz_finished:
            st.markdown("<div style='color:#2980b9; font-size:1.2em; font-weight:bold; margin-top:1em;'>🎉 Quiz complete! Your score: {}/{} </div>".format(st.session_state['score'], len(questions)), unsafe_allow_html=True)

    # --- Flashcards Tab ---
    with tabs[1]:
        show_flashcards()

    # --- Analytics Tab ---
    with tabs[2]:
        show_analytics()

if __name__ == "__main__":
    main()

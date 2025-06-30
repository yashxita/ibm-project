import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# MUST BE FIRST streamlit call
st.set_page_config(page_title="Student Feedback Classifier", page_icon="📚")

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Few-shot template with improved Facilities example
FEW_SHOT_TEMPLATE = """
Classify the following student feedback into EXACTLY one of these three categories: Academics, Facilities, Administration.

You must respond with only one word from: Academics, Facilities, Administration

Examples:
Feedback: "The lectures were very informative and the professor explained clearly."
Category: Academics

Feedback: "The air conditioning in the lecture hall never works."
Category: Facilities

Feedback: "The administration office takes too long to respond to emails."
Category: Administration

Feedback: "The curriculum is updated and fresh."
Category: Academics

Now classify this feedback. Respond only with: Academics, Facilities, or Administration
Feedback: "{feedback}"
Category:"""

# Classification function
def classify_feedback(text):
    prompt = FEW_SHOT_TEMPLATE.format(feedback=text)
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_new_tokens=5)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = prediction[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()

    valid_categories = ["Academics", "Facilities", "Administration"]
    for category in valid_categories:
        if category.lower() in generated_text.lower():
            return category

    # fallback keywords 
    text_lower = text.lower()
    if any(word in text_lower for word in ["library", "cafeteria", "hostel", "wifi", "internet", "building", "room", "facility", "equipment", "ac", "air conditioning", "fan"]):
        return "Facilities"
    elif any(word in text_lower for word in ["admin", "office", "staff", "application", "registration", "procedure", "policy", "management"]):
        return "Administration"
    elif any(word in text_lower for word in ["professor", "lecture", "class", "course", "curriculum", "teachers", "assignment", "exam", "grade"]):
        return "Academics"

    return "Academics"

# ui 
st.title("🎓 Student Feedback Classifier")
st.markdown("Classify feedback into **Academics**, **Facilities**, or **Administration** using FLAN-T5.")

# state keys
if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = ""
if "example_to_use" not in st.session_state:
    st.session_state.example_to_use = ""
if "use_example" not in st.session_state:
    st.session_state.use_example = False
if "clear_trigger" not in st.session_state:
    st.session_state.clear_trigger = False

if st.session_state.use_example:
    st.session_state.feedback_text = st.session_state.example_to_use
    st.session_state.use_example = False
    st.rerun()

if st.session_state.clear_trigger:
    st.session_state.feedback_text = ""
    st.session_state.clear_trigger = False
    st.rerun()

# form
with st.form("classification_form"):
    st.text_area(
        "✍️ Enter student feedback",
        key="feedback_text",
        height=150,
        placeholder="E.g. The internet in the hostel is very slow."
    )
    submit = st.form_submit_button("Classify")

# classified
if submit:
    feedback_input = st.session_state.feedback_text
    if feedback_input.strip():
        with st.spinner("Classifying..."):
            category = classify_feedback(feedback_input)
        st.success(f"🧠 **Predicted Category:** `{category}`")
    else:
        st.warning("Please enter feedback to classify.")

# sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This app uses [FLAN-T5](https://huggingface.co/google/flan-t5-base), a fine-tuned T5 model, to classify student feedback into:
- Academics
- Facilities
- Administration

Built with ❤️ using Streamlit and Hugging Face Transformers.
""")

# example cases
st.sidebar.markdown("Try examples:")
example = st.sidebar.radio(
    "Examples:",
    (
        "The professor explained everything clearly.",
        "The AC in the lecture hall never works.",
        "Admin office lost my application twice.",
    ),
    index=0,
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Use this example"):
        st.session_state.example_to_use = example
        st.session_state.use_example = True
        st.rerun()

with col2:
    if st.button("Clear"):
        st.session_state.clear_trigger = True
        st.rerun()

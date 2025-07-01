import streamlit as st
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Streamlit config
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

# Prompt template
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

# Classifier logic
def classify_feedback(text):
    prompt = FEW_SHOT_TEMPLATE.format(feedback=text)
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_new_tokens=5)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    prediction = prediction.replace("category:", "").strip()

    valid_categories = ["academics", "facilities", "administration"]
    for category in valid_categories:
        if category in prediction:
            return category.capitalize()

    # Fallback
    text_lower = text.lower()
    if any(word in text_lower for word in ["library", "cafeteria", "hostel", "wifi", "internet", "building", "room", "facility", "equipment", "ac", "air conditioning", "fan"]):
        return "Facilities"
    elif any(word in text_lower for word in ["admin", "office", "staff", "application", "registration", "procedure", "policy", "management"]):
        return "Administration"
    elif any(word in text_lower for word in ["professor", "lecture", "class", "course", "curriculum", "teaching", "assignment", "exam", "grade"]):
        return "Academics"

    return "Academics"

# Initialize session state for input text
if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = ""

# UI
st.title("🎓 Student Feedback Classifier")
st.markdown("Classify feedback into **Academics**, **Facilities**, or **Administration** using FLAN-T5.")

option = st.radio("Choose input type:", ["Enter single feedback", "Upload CSV file"])

if option == "Enter single feedback":
    feedback_input = st.text_area(
        "✍️ Enter student feedback", 
        value=st.session_state.feedback_text, 
        height=150, 
        placeholder="E.g. The internet in the hostel is very slow."
    )

    if st.button("Classify"):
        if feedback_input.strip():
            with st.spinner("Classifying..."):
                category = classify_feedback(feedback_input)
            st.success(f"🧠 **Predicted Category:** `{category}`")
        else:
            st.warning("Please enter feedback to classify.")

elif option == "Upload CSV file":
    uploaded_file = st.file_uploader("📁 Upload CSV file with a 'feedback' column", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "feedback" not in df.columns:
                st.error("❌ CSV must contain a 'feedback' column.")
            else:
                with st.spinner("Classifying all feedback entries..."):
                    df["Predicted Category"] = df["feedback"].apply(classify_feedback)

                st.success("✅ Classification completed!")
                st.markdown("### 📊 Results:")
                for i, row in df.iterrows():
                    st.write(f"**Feedback:** {row['feedback']}")
                    st.write(f"🧠 Predicted Category: `{row['Predicted Category']}`")
                    st.markdown("---")

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results CSV", data=csv, file_name="classified_feedback.csv", mime='text/csv')
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Sidebar with examples
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This app uses **[FLAN-T5](https://huggingface.co/google/flan-t5-small)** to classify student feedback into:
- **Academics**
- **Facilities**
- **Administration**

Built with ❤️ using Streamlit and Transformers.
""")

st.sidebar.title("📌 Examples")
example = st.sidebar.radio("Try an example:", [
    "The professor explained everything clearly.",
    "The AC in the lecture hall never works.",
    "Admin office lost my application twice."
])
if st.sidebar.button("Use this example"):
    st.session_state.feedback_text = example
    st.experimental_rerun()

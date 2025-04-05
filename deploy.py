import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import requests

# --- CONFIGURE PAGE ---
st.set_page_config(
    page_title="Buggy Code Detector & Fixer",
    page_icon="🛠️",
    layout="wide"
)

# --- LOAD MODEL AND TOKENIZER ---
model = tf.keras.models.load_model("Trained_model.h5")

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_LENGTH = 1239

HUGGINGFACE_API_KEY = "API KEYS"  

# --- PREPROCESS INPUT FUNCTION ---
def preprocess_input_code(code):
    sequence = tokenizer.texts_to_sequences([code])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    return padded_sequence

# --- GET FIXED CODE FROM HUGGING FACE ---
def get_fixed_code(buggy_code):
    api_url = "https://api-inference.huggingface.co/models/bigcode/starcoder"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"Fix this buggy code:\n```python\n{buggy_code}\n```\nFixed Code:",
        "parameters": {"temperature": 0.3, "max_new_tokens": 200}
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()
        if isinstance(response_data, list):
            fixed_code = response_data[0]["generated_text"].strip()
            return fixed_code
        else:
            return f"⚠ Error: {response_data}"
    except Exception as e:
        return f"⚠ Error generating fix: {str(e)}"

# --- UI HEADER ---
st.title("🛠️ Buggy Code Detector & Auto-Fixer")
st.markdown("### Empowering Developers to Detect & Auto-Fix Bugs with AI 🤖")
st.markdown("---")

# --- MAIN SECTION ---
with st.container():
    col1, col2 = st.columns([2, 1])

    with col1:
        input_code = st.text_area("📝 Paste your Python code here:", height=300, key="unique_code_input")

        uploaded_file = st.file_uploader("📁 Or upload a Python file", type=["py"])
        if uploaded_file:
            input_code = uploaded_file.read().decode("utf-8")
            st.text_area("📄 File Content:", value=input_code, height=300, key="file_code")

    with col2:
        st.markdown("#### 💡 How It Works")
        st.markdown("""
        - 📌 Analyzes your Python code  
        - 🐞 Detects if it's **Buggy** or **Bug-Free**  
        - 🤖 If buggy, then it will auto-suggest a fix  
        - ⚙ Powered by TensorFlow + Hugging Face
        """)

    st.markdown("---")

# --- PREDICTION SECTION ---
if st.button("🚀 Check Code"):
    if input_code.strip():
        with st.spinner("🔍 Analyzing your code..."):
            processed_code = preprocess_input_code(input_code)
            prediction = model.predict(processed_code)

            if prediction[0][0] > 0.5:
                st.error("🐞 Buggy Code Detected!")
                st.info("💡 Generating fixed version...")

                fixed_code = get_fixed_code(input_code)

                st.success("✅ Generated Fixed Code:")
                st.code(fixed_code, language="python")
            else:
                st.success("✅ Your code is Bug-Free! 🎉")
    else:
        st.warning("⚠ Please enter or upload some Python code first!")

# --- FOOTER SECTION ---
st.markdown("---")
st.markdown("#### ✨ Powered by:")
st.markdown("""
- 🧠 **TensorFlow** – Deep Learning Bug Detection  
- 🤗 **Hugging Face StarCoder** – Code Auto-Fixer  
- 🎨 **Streamlit** – Clean UI Framework  
""")

st.markdown(
    "<small>🔒 Secure | 💬 Feedback: [mosespaul@karunya.edu.in](mailto:you@example.com)</small>",
    unsafe_allow_html=True
)

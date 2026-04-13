import streamlit as st
from model import load_model, predict

st.title("🧠 Inappropriate Comments Scanner")

st.write("Detects harmful language including bullying, sarcasm, and implicit bias.")

# Load model once
model, tokenizer, labels = load_model()

user_input = st.text_area("Enter a comment:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        prediction, confidence = predict(user_input, model, tokenizer, labels)
        
        st.subheader("Result:")
        st.write(f"**Category:** {prediction}")
        st.write(f"**Confidence:** {confidence:.2f}")

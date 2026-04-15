import streamlit as st
from model import load_model, predict

st.title("Inappropriate Comments Scanner")
st.write("Detects harmful language including bullying, sarcasm, and harmful content.")
st.title("Inappropriate Comments Scanner")
st.write("Detects harmful language including bullying, sarcasm, and harmful content.")

@st.cache_resource
def get_model():
    return load_model()
@st.cache_resource
def get_model():
    return load_model()

model, tokenizer, labels = get_model()
model, tokenizer, labels = get_model()

user_input = st.text_area("Enter a comment:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        prediction, confidence = predict(user_input, model, tokenizer, labels)


        st.subheader("Result:")
        if confidence < 0.6:
            st.warning(f"**Category:** Uncertain (leaning {prediction})")
            st.write("Low confidence — the comment may be ambiguous or subtly sarcastic.")
        else:
            st.write(f"**Category:** {prediction}")
        st.write(f"**Confidence:** {confidence:.1%}")

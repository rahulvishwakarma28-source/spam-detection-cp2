
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Spam Detection")

@st.cache_resource
def load_model():
    model_path = (
        Path(__file__).resolve().parent
        / "spam_svm_pipeline.joblib"
    )
    return joblib.load(model_path)

model = load_model()

st.title("Spam Detection")
st.write("Enter an SMS message to check whether it is spam or ham.")

with st.form("message_form"):
    message = st.text_area("Enter your message")
    submitted = st.form_submit_button("Check message")

if submitted:
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.warning("Prediction: Spam")
        else:
            st.success("Prediction: Ham (not spam)")

st.caption(
    "This model was tested on SMS messages. "
    "Predictions can be incorrect."
)

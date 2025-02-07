import streamlit as st
import pickle

# Load the saved model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vector = pickle.load(f)

with open("spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📧 Mail Spam Detection System")

user_input = st.text_area("Enter the email message:", "")

if st.button("Predict"):
    if user_input.strip() != "":
        user_vectorized = vector.transform([user_input])
        prediction = model.predict(user_vectorized)
        result = "✅ Ham Mail" if prediction[0] == 1 else "❌ Spam Mail"
        st.subheader(result)
    else:
        st.warning("Please enter a message!")

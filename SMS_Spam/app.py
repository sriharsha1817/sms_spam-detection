import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import streamlit as st
import string

nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Streamlit app
st.set_page_config(page_title="SMS Spam Detection", page_icon="📱", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .header {
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 16px;
        text-align: center;
        color: #333;
        margin-bottom: 40px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        width: 150px;
        height: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.markdown('<div class="header">SMS Spam Detection Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader">Detect if an SMS is Spam or Not Spam using AI-powered models. Enter your text below to check.</div>',
    unsafe_allow_html=True,
)

# Input area
input_sms = st.text_area("Enter the SMS", placeholder="Type your SMS message here...", height=100)

# Predict button
if st.button('Predict'):

    if input_sms.strip():
        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        # Vectorize the input
        vector_input = tk.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]

        # Display the result with color-coded output
        if result == 1:
            st.markdown('<h3 style="color:red; text-align:center;">🚨 Spam Message 🚨</h3>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 style="color:green; text-align:center;">✅ Not Spam ✅</h3>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid SMS message before clicking Predict.")

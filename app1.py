import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")
    st.stop()

ps = PorterStemmer()

def transform_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
        
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

# Load the saved models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' exist.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms:
        st.warning("Please enter a message to classify.")
    else:
        try:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            st.write("Transformed text:", transformed_sms)
            
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            
            # 3. predict
            result = model.predict(vector_input)[0]
            
            # 4. Display
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Please try again with a different message.")

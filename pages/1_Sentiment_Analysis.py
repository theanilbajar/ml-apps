import streamlit as st
from transformers import pipeline
import time

st.set_page_config(page_title="Sentiment Analysis")

st.markdown("# Sentiment Analysis with HuggingFace")
st.sidebar.header("Sentiment Analysis with HuggingFace")
st.write(
    """Sentiment Analysis with HuggingFace and streamlit"""
)

start_time = time.time()

@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis')

model = load_model()

end_time = time.time()
st.write(f'Model loading took {round(end_time-start_time, 5)} seconds')

query = st.text_area("your query", value='I like NLP', height=5)

st.write('### Result: ')
if query:
    result = model(query)[0]
    st.write(f"Sentiment: {result.get('label')}")
    st.write(f"Sentiment Confidence: {round(result.get('score'), 6)}")


import streamlit as st
from transformers import pipeline
from transformers import pipeline, AutoTokenizer, TFAutoModelForTokenClassification

st.set_page_config(page_title="Named Entity Recognition")

st.markdown("# Named Entity Recognition with BERT")
st.sidebar.header("Named Entity Recognition with BERT")
st.write(
    """This application show usage of BERT models for Named Entity Recognition."""
)

# load the bert model for NER
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = TFAutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# create pipeline
pipe = pipeline("ner", model=model, tokenizer=tokenizer)

query = st.text_area("your query", "My name is Clara and I live in Berkeley, California.")

# Function to visualize NER entities in the text
def visualize_entities(text, entities):
    current_position = 0
    result = ""

    for entity in entities:
        start, end = entity['start'], entity['end']
        entity_label = entity['entity']
        result += text[current_position:start]  # Add text before the entity
        result += f'<span style="color: black; background-color: #ec4899;">[{entity_label}: {text[start:end]}]</span>' 
        # Highlight the entity
        current_position = end

    # Add any remaining text after the last entity
    result += text[current_position:]

    st.markdown(result, unsafe_allow_html = True)

if query:
    ner_results = pipe(query)

    st.write("### Result: ")

    # Visualize the entities in the text
    visualize_entities(query, ner_results)



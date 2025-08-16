import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load models
EN_FR_MODEL = "niharikabhardwaj/en-fr-model"
EN_HI_MODEL = "niharikabhardwaj/en-hi-model"

@st.cache_resource
def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

en_fr_tokenizer, en_fr_model = load_model(EN_FR_MODEL)
en_hi_tokenizer, en_hi_model = load_model(EN_HI_MODEL)

# Streamlit UI
st.set_page_config(page_title="Dual Language Translator", page_icon="🌍")
st.title("🌍 English ↔ Frech/Hindi Translator")

direction = st.radio("Select translation direction:", ["English → French", "English → Hindi"])
text = st.text_area("Enter text:")

if st.button("Translate"):
    if text.strip():
        num_words = len(text.split())
        num_letters = len([c for c in text if c.isalpha()])

        if num_words >= 10 or num_letters >= 10:
            if direction == "English → French":
                inputs = en_fr_tokenizer(text, return_tensors="pt", padding=True)
                translated = en_fr_model.generate(**inputs)
                output = en_fr_tokenizer.decode(translated[0], skip_special_tokens=True)
                st.success(output)
            else:
                inputs = en_hi_tokenizer(text, return_tensors="pt", padding=True)
                translated = en_hi_model.generate(**inputs)
                output = en_hi_tokenizer.decode(translated[0], skip_special_tokens=True)
                st.success(output)
        else:
            st.warning("Please enter at least 10 words or 10 letters to translate!")
    else:
        st.warning("Please enter some text to translate!")


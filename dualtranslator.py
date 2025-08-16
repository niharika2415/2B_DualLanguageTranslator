import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load models
EN_FR_MODEL = "niharikabhardwaj/en-fr-model"
HI_EN_MODEL = "niharikabhardwaj/en-hi-model"

@st.cache_resource
def load_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

en_fr_tokenizer, en_fr_model = load_model(EN_FR_MODEL)
fr_en_tokenizer, fr_en_model = load_model(FR_EN_MODEL)

# Streamlit UI
st.set_page_config(page_title="Dual Language Translator", page_icon="🌍")
st.title("🌍 English ↔ French Translator")

direction = st.radio("Select translation direction:", ["English → French", "French → English"])
text = st.text_area("Enter text:")

if st.button("Translate"):
    if len(text) > 10:
        if text.strip():
            if direction == "English → French":
                inputs = en_fr_tokenizer(text, return_tensors="pt", padding=True)
                translated = en_fr_model.generate(**inputs)
                output = en_fr_tokenizer.decode(translated[0], skip_special_tokens=True)
                st.success(output)
            else:
                inputs = fr_en_tokenizer(text, return_tensors="pt", padding=True)
                translated = fr_en_model.generate(**inputs)
                output = fr_en_tokenizer.decode(translated[0], skip_special_tokens=True)
                st.success(output)
        else:
            st.warning("Please enter some text to translate!")
            if direction == "English → Hindi":
                inputs = en_hi_tokenizer(text, return_tensors="pt", padding=True)
                translated = en_hi_model.generate(**inputs)
                output = en_hi_tokenizer.decode(translated[0], skip_special_tokens=True)
                st.success(output)
            else:
                inputs = hi_en_tokenizer(text, return_tensors="pt", padding=True)
                translated = hi_en_model.generate(**inputs)
                output = hi_en_tokenizer.decode(translated[0], skip_special_tokens=True)
                st.success(output)
    else:
        st.warning("Please enter some text to translate!")    


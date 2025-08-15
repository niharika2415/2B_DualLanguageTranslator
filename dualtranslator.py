import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import requests
import zipfile
import io

# --- Helper Functions ---

@st.cache_resource
def load_models():
    """
    Loads fine-tuned translation models from a public Google Drive URL.
    This function caches the models to avoid downloading them on each interaction.
    """
    st.info("Loading fine-tuned translation models. This may take a moment...")

    # The Google Drive URLs for fine-tuned model folders (zipped).

    drive_url_fr = "https://drive.google.com/file/d/1dWe1J36dzdBAho4x_LRPfVFiuumyOi3n/view?usp=sharing"
    drive_url_hi = "https://drive.google.com/file/d/1Tey7cHou8q0D_40Q5fk6tpqeROO3ZR3h/view?usp=sharing"
    
    # Check if the model folders already exist to avoid re-downloading
    if not os.path.exists("./fine-tuned-en-fr-model"):
        st.info("Downloading French model from Google Drive...")
        try:
            response = requests.get(drive_url_fr)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall("./fine-tuned-en-fr-model")
            st.success("French model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading French model from Drive: {e}")
            return None

    if not os.path.exists("./fine-tuned-en-hi-model"):
        st.info("Downloading Hindi model from Google Drive...")
        try:
            response = requests.get(drive_url_hi)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall("./fine-tuned-en-hi-model")
            st.success("Hindi model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading Hindi model from Drive: {e}")
            return None

    # Load the models from the local directories after downloading
    try:
        tokenizer_en_fr = AutoTokenizer.from_pretrained("./fine-tuned-en-fr-model")
        model_en_fr = AutoModelForSeq2SeqLM.from_pretrained("./fine-tuned-en-fr-model")

        tokenizer_en_hi = AutoTokenizer.from_pretrained("./fine-tuned-en-hi-model")
        model_en_hi = AutoModelForSeq2SeqLM.from_pretrained("./fine-tuned-en-hi-model")

        st.success("Models loaded successfully!")
        return {
            "fr_tokenizer": tokenizer_en_fr,
            "fr_model": model_en_fr,
            "hi_tokenizer": tokenizer_en_hi,
            "hi_model": model_en_hi
        }
    except Exception as e:
        st.error(f"Error loading models from local disk: {e}")
        return None

def translate_text(text, tokenizer, model):
    """
    Translates a single piece of text using the given tokenizer and model.
    """
    if not text:
        return ""
    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    # Generate the translation
    outputs = model.generate(input_ids)
    
    # Decode the output and return the translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# --- Main Streamlit Application ---

def main():
    st.set_page_config(page_title="Dual Language Translator", page_icon="🌐")
    st.title("🌐 English to French & Hindi Translator")
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stTextInput>div>div>input {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("Enter an English word or line with **10 or more letters** to see its translation.")

    # Load models into cache
    models = load_models()
    if models is None:
        st.stop()

    # Get user input
    user_input = st.text_input("Enter English text here:")

    if st.button("Translate"):
        if len(user_input) < 10:
            st.warning("Please upload again. The English word/line must have 10 or more letters.")
        else:
            with st.spinner("Translating..."):
                # Translate to French
                fr_output = translate_text(user_input, models["fr_tokenizer"], models["fr_model"])
                
                # Translate to Hindi
                hi_output = translate_text(user_input, models["hi_tokenizer"], models["hi_model"])

                st.subheader("French Translation:")
                st.write(fr_output)

                st.subheader("Hindi Translation:")
                st.write(hi_output)

if __name__ == "__main__":
    main()

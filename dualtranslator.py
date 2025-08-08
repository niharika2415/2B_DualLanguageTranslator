import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown
import os

# --- Streamlit Page Configuration ---
# It's best practice to set page config at the very top.
st.set_page_config(page_title="Dual Translator", layout="centered")

# --- UI Elements for User Input ---
# All UI elements are now in a single, clean block to avoid duplication.
st.title("🔁 English ➡️ French & Hindi Translator")
st.markdown("Enter an English sentence (10+ letters). You'll get translations in both languages.")
user_input = st.text_input("📥 Your English sentence:")

# --- Define Max Sequence Lengths (IMPORTANT: YOU MUST REPLACE THESE VALUES!) ---
# These are the placeholder values from your notebook. You need to replace them
# with the exact numbers you get from your notebook's output.
max_len_eng_fr = 4
max_len_fr = 10
max_len_eng_hi = 22
max_len_hi = 25

# --- Model and Tokenizer Loading (cached for efficiency) ---
# Using st.cache_resource ensures that these heavy files are only loaded once.
@st.cache_resource
def load_translation_resources():
    # French model
    fr_model_id = "1xdvwtu6Js8Vt8wSlU4l0bmbcehpSowD4"
    fr_model_path = "english_to_french_model.h5"
    if not os.path.exists(fr_model_path):
        gdown.download(f"https://drive.google.com/uc?id={fr_model_id}", fr_model_path, quiet=False)
    model_fr = tf.keras.models.load_model(fr_model_path)

    # Hindi model
    hi_model_id = "1oPZ0a_jbfG_-YsUuefyx6fK-kJ1mmqXj"
    hi_model_path = "english_to_hindi_model.h5"
    if not os.path.exists(hi_model_path):
        gdown.download(f"https://drive.google.com/uc?id={hi_model_id}", hi_model_path, quiet=False)
    model_hi = tf.keras.models.load_model(hi_model_path)

    # 1. English tokenizer for French model (using ID)
    eng_tokenizer_fr_path = "eng_tokenizer_fr.pkl"
    eng_tokenizer_fr_id = "1mEzR2ePmY0zbzHHPlMwNe9WofMK-vKUY"
    if not os.path.exists(eng_tokenizer_fr_path):
        gdown.download(id=eng_tokenizer_fr_id, output=eng_tokenizer_fr_path, quiet=False)
    try:
        with open(eng_tokenizer_fr_path, "rb") as f:
            eng_tokenizer_fr = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading {eng_tokenizer_fr_path}: {e}")
        st.stop()

    # 2. French tokenizer (using ID)
    fr_tokenizer_id = "1t2SERaR-ugYvf49eGbQKwdW0qDKRR7aF"
    fr_tokenizer_path = "fr_tokenizer.pkl"
    if not os.path.exists(fr_tokenizer_path):
        gdown.download(id=fr_tokenizer_id, output=fr_tokenizer_path, quiet=False)
    try:
        with open(fr_tokenizer_path, "rb") as f:
            fr_tokenizer = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading {fr_tokenizer_path}: {e}")
        st.stop()

    # 3. English tokenizer for Hindi model (using ID)
    eng_tokenizer_hi_path = "eng_tokenizer_hi.pkl"
    eng_tokenizer_hi_id = "1WxdefG3TY9uJkpt5yfqxwqQ-U9czudlr"
    if not os.path.exists(eng_tokenizer_hi_path):
        gdown.download(id=eng_tokenizer_hi_id, output=eng_tokenizer_hi_path, quiet=False)
    try:
        with open(eng_tokenizer_hi_path, "rb") as f:
            eng_tokenizer_hi = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading {eng_tokenizer_hi_path}: {e}")
        st.stop()

    # 4. Hindi tokenizer (using ID)
    hi_tokenizer_path = "hi_tokenizer.pkl"
    hi_tokenizer_id = "1vAbYqb1X2PxXHZwCT2d_8PKG-P8TDONm"
    if not os.path.exists(hi_tokenizer_path):
        gdown.download(id=hi_tokenizer_id, output=hi_tokenizer_path, quiet=False)
    try:
        with open(hi_tokenizer_path, "rb") as f:
            hi_tokenizer = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading {hi_tokenizer_path}: {e}")
        st.stop()

    return model_fr, model_hi, eng_tokenizer_fr, fr_tokenizer, eng_tokenizer_hi, hi_tokenizer

# Load all resources
model_fr, model_hi, eng_tokenizer_fr, fr_tokenizer, eng_tokenizer_hi, hi_tokenizer = load_translation_resources()

# --- Translation function (corrected) ---
# max_output_len parameter removed as it's not used in the function logic
def translate_sentence(model, input_text, tokenizer_in, tokenizer_out, max_input_len):
    seq = tokenizer_in.texts_to_sequences([input_text])
    padded = pad_sequences(seq, maxlen=max_input_len, padding='post')
    pred = model.predict(padded)
    pred_seq = np.argmax(pred, axis=-1)

    result = []
    # Create a reverse word index for faster lookup
    reverse_word_index = {index: word for word, index in tokenizer_out.word_index.items()}

    for idx in pred_seq[0]:
        if idx == 0:
            continue
        word = reverse_word_index.get(idx)
        if word:
            result.append(word)
    return ' '.join(result)

# --- Translate Button and Output ---
if st.button("Translate"):
    if len(user_input.strip()) < 10:
        st.warning("⚠️ Enter at least 10 letters.")
    else:
        # French Translation (using the correct max length for this model)
        fr_translation = translate_sentence(
            model_fr,
            user_input.lower(),
            eng_tokenizer_fr,
            fr_tokenizer,
            max_input_len=max_len_eng_fr,
        )
        st.markdown(f"**🇫🇷 French:** {fr_translation}")

        # Hindi Translation (using the correct max length for this model)
        hi_translation = translate_sentence(
            model_hi,
            user_input.lower(),
            eng_tokenizer_hi,
            hi_tokenizer,
            max_input_len=max_len_eng_hi,
        )
        st.markdown(f"**🇮🇳 Hindi:** {hi_translation}")

        st.success("✅ Translations Complete!")

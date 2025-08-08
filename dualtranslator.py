import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown
import os

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Dual Translator", layout="centered")

# --- UI Elements for User Input ---
st.title("🔁 English ➡️ French & Hindi Translator")
st.markdown("Enter an English sentence (10+ letters). You'll get translations in both languages.")
user_input = st.text_input("📥 Your English sentence:")

# --- Define Max Sequence Lengths (IMPORTANT: Replace with your actual values!) ---
max_len_eng = 50  # Example: Adjust based on your English tokenizer/model's max sequence length
max_len_fr = 60   # Example: Adjust based on your French tokenizer/model's max sequence length
max_len_hi = 60   # Example: Adjust based on your Hindi tokenizer/model's max sequence length

# --- Model and Tokenizer Loading (with gdown 'id' method and error handling) ---

@st.cache_resource # Cache resource to prevent reloading on every rerun
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

    # 1. English tokenizer for French model
    eng_tokenizer_fr_path = "eng_tokenizer_fr.pkl"
    eng_tokenizer_fr_id = "1mEzR2ePmY0zbzHHPlMwNe9WofMK-vKUY" # Corrected to use ID
    if not os.path.exists(eng_tokenizer_fr_path):
        gdown.download(id=eng_tokenizer_fr_id, output=eng_tokenizer_fr_path, quiet=False)
    try:
        with open(eng_tokenizer_fr_path, "rb") as f:
            eng_tokenizer_fr = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading {eng_tokenizer_fr_path}: {e}")
        st.error("Please ensure the file is a valid pickle file and was downloaded correctly.")
        st.stop() # Stop the app if critical component fails to load

    # 2. French tokenizer
    fr_tokenizer_id = "1t2SERaR-ugYvf49eGbQKwdW0qDKRR7aF"
    fr_tokenizer_path = "fr_tokenizer.pkl"
    if not os.path.exists(fr_tokenizer_path):
        gdown.download(id=fr_tokenizer_id, output=fr_tokenizer_path, quiet=False)
    try:
        with open(fr_tokenizer_path, "rb") as f:
            fr_tokenizer = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading {fr_tokenizer_path}: {e}")
        st.error("Please ensure the file is a valid pickle file and was downloaded correctly.")
        st.stop()

    # 3. English tokenizer for Hindi model
    eng_tokenizer_hi_path = "eng_tokenizer_hi.pkl"
    eng_tokenizer_hi_id = "1WxdefG3TY9uJkpt5yfqxwqQ-U9czudlr" # Corrected to use ID
    if not os.path.exists(eng_tokenizer_hi_path):
        gdown.download(id=eng_tokenizer_hi_id, output=eng_tokenizer_hi_path, quiet=False)
    try:
        with open(eng_tokenizer_hi_path, "rb") as f:
            eng_tokenizer_hi = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading {eng_tokenizer_hi_path}: {e}")
        st.error("Please ensure the file is a valid pickle file and was downloaded correctly.")
        st.stop()

    # 4. Hindi tokenizer
    hi_tokenizer_path = "hi_tokenizer.pkl"
    hi_tokenizer_id = "1vAbYqb1X2PxXHZwCT2d_8PKG-P8TDONm" # Corrected to use ID
    if not os.path.exists(hi_tokenizer_path):
        gdown.download(id=hi_tokenizer_id, output=hi_tokenizer_path, quiet=False)
    try:
        with open(hi_tokenizer_path, "rb") as f:
            hi_tokenizer = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading {hi_tokenizer_path}: {e}")
        st.error("Please ensure the file is a valid pickle file and was downloaded correctly.")
        st.stop()

    return model_fr, model_hi, eng_tokenizer_fr, fr_tokenizer, eng_tokenizer_hi, hi_tokenizer

# Load all resources (models and tokenizers) using the cached function
model_fr, model_hi, eng_tokenizer_fr, fr_tokenizer, eng_tokenizer_hi, hi_tokenizer = load_translation_resources()

# --- Translation function ---
def translate_sentence(model, input_text, tokenizer_in, tokenizer_out, max_input_len, max_output_len):
    seq = tokenizer_in.texts_to_sequences([input_text])
    padded = pad_sequences(seq, maxlen=max_input_len, padding='post')
    pred = model.predict(padded)
    pred_seq = np.argmax(pred, axis=-1)

    result = []
    # Create a reverse word index for faster lookup
    reverse_word_index = {index: word for word, index in tokenizer_out.word_index.items()}

    for idx in pred_seq[0]:
        if idx == 0: # Assuming 0 is padding, adjust if your tokenizer uses a different padding index
            continue
        word = reverse_word_index.get(idx)
        if word: # Check if word exists (handles OOV or unknown tokens)
            result.append(word)
    return ' '.join(result)

# --- Translate Button and Output ---
if st.button("Translate"):
    if len(user_input.strip()) < 10:
        st.warning("⚠️ Enter at least 10 letters.")
    else:
        # French Translation
        fr_translation = translate_sentence(
            model_fr,
            user_input.lower(),
            eng_tokenizer_fr,
            fr_tokenizer,
            max_input_len=max_len_eng,
            max_output_len=max_len_fr # max_output_len is not used in translate_sentence, can remove from signature if not needed
        )
        st.markdown(f"**🇫🇷 French:** {fr_translation}")

        # Hindi Translation
        hi_translation = translate_sentence(
            model_hi,
            user_input.lower(),
            eng_tokenizer_hi,
            hi_tokenizer,
            max_input_len=max_len_eng,
            max_output_len=max_len_hi # max_output_len is not used
        )
        st.markdown(f"**🇮🇳 Hindi:** {hi_translation}")

        st.success("✅ Translations Complete!") # Moved success message to the end

import streamlit as st 
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequence
import gdown
import pickle

#Title
st.title("Dual Language Translator")
st.write("Enter an English sentence (10K+ letters) to get translations in French and Hindi.")

#Input
user_input= st.text_input("Enter An English Sentence:")

if user_input:
    if len(user_input) < 10:
        st.warning("Upload again")
    else:
        st.success("Valid Input! Translating...")

# French model
fr_model_id = "1xdvwtu6Js8Vt8wSlU4l0bmbcehpSowD4"  # Drive ID
fr_model_path = "english_to_french_model.h5"
if not os.path.exists(fr_model_path):
    gdown.download(f"https://drive.google.com/uc?id={fr_model_id}", fr_model_path, quiet=False)
model_fr = tf.keras.models.load_model(fr_model_path)

# Hindi model
hi_model_id = "1oPZ0a_jbfG_-YsUuefyx6fK-kJ1mmqXj"  # Drive ID
hi_model_path = "english_to_hindi_model.h5"
if not os.path.exists(hi_model_path):
    gdown.download(f"https://drive.google.com/uc?id={hi_model_id}", hi_model_path, quiet=False)
model_hi = tf.keras.models.load_model(hi_model_path)

# 1. English tokenizer for French model
eng_tokenizer_fr_path = "eng_tokenizer_fr.pkl"
eng_tokenizer_fr_url = "https://drive.google.com/file/d/1mEzR2ePmY0zbzHHPlMwNe9WofMK-vKUY/view?usp=sharing"
if not os.path.exists(eng_tokenizer_fr_path):
    gdown.download(eng_tokenizer_fr_url, eng_tokenizer_fr_path, quiet=False)
with open(eng_tokenizer_fr_path, "rb") as f:
    eng_tokenizer_fr = pickle.load(f)

# 2. French tokenizer
fr_tokenizer_path = "fr_tokenizer.pkl"
fr_tokenizer_url = "https://drive.google.com/file/d/1t2SERaR-ugYvf49eGbQKwdW0qDKRR7aF/view?usp=sharing"
if not os.path.exists(fr_tokenizer_path):
    gdown.download(fr_tokenizer_url, fr_tokenizer_path, quiet=False)
with open(fr_tokenizer_path, "rb") as f:
    fr_tokenizer = pickle.load(f)

# 3. English tokenizer for Hindi model
eng_tokenizer_hi_path = "eng_tokenizer_hi.pkl"
eng_tokenizer_hi_url = "https://drive.google.com/file/d/1WxdefG3TY9uJkpt5yfqxwqQ-U9czudlr/view?usp=sharing"
if not os.path.exists(eng_tokenizer_hi_path):
    gdown.download(eng_tokenizer_hi_url, eng_tokenizer_hi_path, quiet=False)
with open(eng_tokenizer_hi_path, "rb") as f:
    eng_tokenizer_hi = pickle.load(f)

# 4. Hindi tokenizer
hi_tokenizer_path = "hi_tokenizer.pkl"
hi_tokenizer_url = "https://drive.google.com/file/d/1vAbYqb1X2PxXHZwCT2d_8PKG-P8TDONm/view?usp=sharing"
if not os.path.exists(hi_tokenizer_path):
    gdown.download(hi_tokenizer_url, hi_tokenizer_path, quiet=False)
with open(hi_tokenizer_path, "rb") as f:
    hi_tokenizer = pickle.load(f)

# Translation function
def translate_sentence(model, input_text, tokenizer_in, tokenizer_out, max_input_len, max_output_len):
    seq = tokenizer_in.texts_to_sequences([input_text])
    padded = pad_sequences(seq, maxlen=max_input_len, padding='post')
    pred = model.predict(padded)
    pred_seq = np.argmax(pred, axis=-1)

    result = []
    for idx in pred_seq[0]:
        if idx == 0:
            continue
        for word, index in tokenizer_out.word_index.items():
            if index == idx:
                result.append(word)
                break
    return ' '.join(result)

# Streamlit UI
st.set_page_config(page_title="Dual Translator", layout="centered")
st.title("🔁 English ➡️ French & Hindi Translator")
st.markdown("Enter an English sentence (10+ letters). You'll get translations in both languages.")

user_input = st.text_input("📥 Your English sentence:")

if st.button("Translate"):
    if len(user_input.strip()) < 10:
        st.warning("⚠️ Enter at least 10 letters.")
    else:
        # French
        fr_translation = translate_sentence(
            model_fr,
            user_input.lower(),
            eng_tokenizer_fr,
            fr_tokenizer,
            max_input_len=max_len_eng,
            max_output_len=max_len_fr
        )

        # Hindi
        hi_translation = translate_sentence(
            model_hi,
            user_input.lower(),
            eng_tokenizer_hi,
            hi_tokenizer,
            max_input_len=max_len_eng,
            max_output_len=max_len_hi
        )

        st.success("✅ Translations")
        st.markdown(f"**🇫🇷 French:** {fr_translation}")
        st.markdown(f"**🇮🇳 Hindi:** {hi_translation}")
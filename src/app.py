import streamlit as st
import tensorflow as tf
import numpy as np
from model import TransformerEncoder, PositionalEmbedding, TransformerDecoder
from custom_tokenizer import custom_standardization

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models/seq2seq-transformer_full_vocab.h5',
    custom_objects={
        "TransformerEncoder" : TransformerEncoder,
        "PositionalEmbedding" : PositionalEmbedding,
        "TransformerDecoder" : TransformerDecoder
    })

    return model

def load_tokenizer():
    eng_token = tf.keras.models.load_model('tokenizer/eng_token_layer')
    deu_token = tf.keras.models.load_model('tokenizer/deu_token_layer')

    return eng_token, deu_token

with st.spinner('Loading model into Memory...'):
    model = load_model()
    eng_token, deu_token = load_tokenizer()

# Creating sequence
deu_vocab = deu_token.layers[0].get_vocabulary()
deu_index_lookup = dict(zip(range(len(deu_vocab)), deu_vocab))
max_decode_sentence_length = 30

def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_token.predict([input_sentence]).to_tensor(default_value=0, shape=[None, max_decode_sentence_length])
    decode_sentence = "[start]"
    for i in range(max_decode_sentence_length):
        tokenized_target_sentence = deu_token.predict([decode_sentence]).to_tensor(default_value=0, shape=[None, max_decode_sentence_length+1])[:, :-1]
        predictions = model([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = deu_index_lookup[sampled_token_index]
        decode_sentence += " " + sampled_token
        
        if sampled_token == "[end]":
            break
    return decode_sentence


# Main Body
st.title("Translator 🇬🇧 -> 🇩🇪")
col1, col2 = st.columns(2)

with col1:
    option_source = st.selectbox('Select source language',
    ('English', 'German'))

    input_text = st.text_input("Input the text you want to translate")


with col2:
    option_target = st.selectbox('Select target language',
    ('German', 'English'))
    if input_text == "":
        st.text_area('Translated text')
    else:
        translated_text = decode_sequence(input_text)
        st.text_area('Translated text', translated_text[7:-5])
    


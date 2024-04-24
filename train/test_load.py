"""
TODO: delete this file after finishing the main part of the project
"""
import os

import tensorflow as tf
import numpy as np

from en_ge_train import create_architecture_en_ge, compile_model
from ge_en_train import create_architecture_ge_en


vocab_size = 3000
max_length = 50
text_vec_layer_en = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length)
text_vec_layer_de = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length)

# Load the vocabulary from the .txt file
with open(os.path.join(os.getcwd(), "train/vocabularies_de_en/vocab_en.txt"), 'r') as f:
    vocabulary_en = [line.strip() for line in f]

text_vec_layer_en.set_vocabulary(vocabulary_en)

with open(os.path.join(os.getcwd(), "train/vocabularies_de_en/vocab_de.txt"), 'r') as f:
    vocabulary_de = [line.strip() for line in f]

text_vec_layer_de.set_vocabulary(vocabulary_de)


model2 = create_architecture_ge_en()

model2.load_weights(os.path.join(os.getcwd(), "models/german-to-english/de_en/de_en"))

model2 = compile_model(model2)

print("ok")


def translate_en_ge(sentence, model, max_length=50):
    translation = ""

    for word_idx in range(max_length):
        X = np.array([sentence])  # encoder input
        X_dec = np.array(["startofseq " + translation])  # decoder input
        y_proba = model.predict((X, X_dec))[0, word_idx]  # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = text_vec_layer_de.get_vocabulary()[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()


def translate_ge_en(sentence, model, max_length=50):
    translation = ""

    for word_idx in range(max_length):
        X = np.array([sentence])  # encoder input
        X_dec = np.array(["startofseq " + translation])  # decoder input
        y_proba = model.predict((X, X_dec))[0, word_idx]  # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = text_vec_layer_en.get_vocabulary()[predicted_word_id]

        if predicted_word == "endofseq":
            break

        translation += " " + predicted_word

    return translation.strip()

answ = translate_ge_en("Ist er Tom?", model2)

print(answ)

# answ = translate_en_ge("i am ready.", model)

# print(answ)
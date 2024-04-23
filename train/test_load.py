"""
TODO: delete this file after finishing the main part of the project
"""
import os

import tensorflow as tf
import numpy as np

from en_ge_train import create_architecture, compile_model


vocab_size = 3000
max_length = 50
text_vec_layer_en = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length)
text_vec_layer_de = tf.keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length)

# Load the vocabulary from the .txt file
with open(os.path.join(os.getcwd(), "train/vocabularies/vocab_en.txt"), 'r') as f:
    vocabulary_en = [line.strip() for line in f]

text_vec_layer_en.set_vocabulary(vocabulary_en)

with open(os.path.join(os.getcwd(), "train/vocabularies/vocab_de.txt"), 'r') as f:
    vocabulary_de = [line.strip() for line in f]

text_vec_layer_de.set_vocabulary(vocabulary_de)

model = create_architecture()

model.load_weights(os.path.join(os.getcwd(), "models/english-to-german/en_de/en_de"))

compile_model(model)

print("ok")

def translate(sentence, model, max_length=50):
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

answ = translate("he is married to his wife", model)
print(answ)
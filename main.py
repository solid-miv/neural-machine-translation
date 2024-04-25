import os
import tkinter as tk
from tkinter import messagebox

import tensorflow as tf
import numpy as np

from train.en_ge_train import create_architecture_en_ge, compile_model
from train.ge_en_train import create_architecture_ge_en

# constant
VOCAB_SIZE = 3000
MAX_LENGTH = 50

# instantiate the text vectorization layers
text_vec_layer_de = tf.keras.layers.TextVectorization(VOCAB_SIZE, output_sequence_length=MAX_LENGTH)
with open(os.path.join(os.getcwd(), "train/vocabularies_en_de/vocab_de.txt"), 'r') as f:
    vocabulary_de = [line.strip() for line in f]
text_vec_layer_de.set_vocabulary(vocabulary_de)

text_vec_layer_en = tf.keras.layers.TextVectorization(VOCAB_SIZE, output_sequence_length=MAX_LENGTH)
with open(os.path.join(os.getcwd(), "train/vocabularies_de_en/vocab_en.txt"), 'r') as f:
    vocabulary_en = [line.strip() for line in f]
text_vec_layer_en.set_vocabulary(vocabulary_en)

# instantiate the models
model_en_ge = create_architecture_en_ge()
model_en_ge.load_weights(os.path.join(os.getcwd(), "models/english-to-german/en_de/en_de"))
model_en_ge = compile_model(model_en_ge)

model_ge_en = create_architecture_ge_en()
model_ge_en.load_weights(os.path.join(os.getcwd(), "models/german-to-english/de_en/de_en"))
model_ge_en = compile_model(model_ge_en)


def translate_en_ge(sentence, model=model_en_ge, max_length=MAX_LENGTH):
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


def translate_ge_en(sentence, model=model_ge_en, max_length=50):
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


def eng_to_ger():
    text = eng_text.get("1.0", "end-1c")
    if text == "":
        messagebox.showwarning("Warning", "Please, enter the text to translate")
    else:
        # Call your translation function here
        translation = translate_en_ge(text)
        ger_translation.delete("1.0", tk.END)
        ger_translation.insert(tk.END, translation)


def translate_ger_to_eng():
    text = ger_text.get("1.0", "end-1c")
    if text == "":
        messagebox.showwarning("Warning", "Please, enter the text to translate")
    else:
        # Call your translation function here
        translation = translate_ge_en(text)
        eng_translation.delete("1.0", tk.END)
        eng_translation.insert(tk.END, translation)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Neural Machine Translator")

    tk.Label(root, text="English Text:", font=("Arial", 10)).grid(row=0, column=0)
    eng_text = tk.Text(root, height=5, width=50)
    eng_text.grid(row=1, column=0)

    btn_eng_to_ger = tk.Button(root, text="Translate English to German", command=eng_to_ger)
    btn_eng_to_ger.grid(row=2, column=0)

    ger_translation = tk.Text(root, height=5, width=50)
    ger_translation.grid(row=3, column=0)

    tk.Label(root, text="German Text:", font=("Arial", 10)).grid(row=0, column=1)
    ger_text = tk.Text(root, height=5, width=50)
    ger_text.grid(row=1, column=1)

    btn_ger_to_eng = tk.Button(root, text="Translate German to English", command=translate_ger_to_eng)
    btn_ger_to_eng.grid(row=2, column=1)

    eng_translation = tk.Text(root, height=5, width=50)
    eng_translation.grid(row=3, column=1)

    root.mainloop()
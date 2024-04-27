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


def create_text_layer_de():
    """
    Creates a text vectorization layer for the German language.

    Returns:
        text_vec_layer_de (tf.keras.layers.TextVectorization): The text vectorization layer for German.
    """
    text_vec_layer_de = tf.keras.layers.TextVectorization(VOCAB_SIZE, output_sequence_length=MAX_LENGTH)

    with open(os.path.join(os.getcwd(), "train/vocabularies_en_de/vocab_de.txt"), 'r', encoding="utf-8") as f:
        vocabulary_de = [line.strip() for line in f]

    text_vec_layer_de.set_vocabulary(vocabulary_de)

    return text_vec_layer_de


def create_text_layer_en():
    """
    Creates a text vectorization layer for English text.

    Returns:
        text_vec_layer_en (tf.keras.layers.TextVectorization): The text vectorization layer for English text.
    """
    text_vec_layer_en = tf.keras.layers.TextVectorization(VOCAB_SIZE, output_sequence_length=MAX_LENGTH)

    with open(os.path.join(os.getcwd(), "train/vocabularies_de_en/vocab_en.txt"), 'r', encoding="utf-8") as f:
        vocabulary_en = [line.strip() for line in f]

    text_vec_layer_en.set_vocabulary(vocabulary_en)

    return text_vec_layer_en


def create_model_en_ge():
    """
    Creates and returns a model for English-to-German translation.

    Returns:
        model_en_ge (Model): The created model for English-to-German translation.
    """
    model_en_ge = create_architecture_en_ge()
    model_en_ge.load_weights(os.path.join(os.getcwd(), "models/english-to-german/en_de/en_de"))
    model_en_ge = compile_model(model_en_ge)

    return model_en_ge


def create_model_ge_en():
    """
    Creates and returns a model for German-to-English translation.

    Returns:
        model_ge_en (Model): The compiled model for German-to-English translation.
    """
    model_ge_en = create_architecture_ge_en()
    model_ge_en.load_weights(os.path.join(os.getcwd(), "models/german-to-english/de_en/de_en"))
    model_ge_en = compile_model(model_ge_en)

    return model_ge_en


def translate_en_ge(sentence, model, text_vec_layer_de, max_length=MAX_LENGTH):
    """
    Translates an English sentence to German using a given model.

    Parameters:
        sentence (str): The English sentence to be translated.
        model (tf.keras.Model): The translation model to be used. 
        text_vec_layer_de (tf.keras.layers.TextVectorization): The text vectorization layer for German. 
        max_length (int, optional): The maximum length of the translated sentence. Defaults to MAX_LENGTH constant.

    Returns:
        str: The translated German sentence.
    """
    translation = ""
    punctuation = ""

    # extracting punctuation from the sentence
    if sentence[-1] in ['.', '?', '!']:
        punctuation = sentence[-1]
        sentence = sentence[:-1]

    for word_idx in range(max_length):
        X = np.array([sentence])  # encoder input
        X_dec = np.array(["startofseq " + translation])  # decoder input
        y_proba = model.predict((X, X_dec))[0, word_idx]  # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = text_vec_layer_de.get_vocabulary()[predicted_word_id]

        if predicted_word == "endofseq":
            break

        translation += " " + predicted_word

    translation += punctuation

    return translation.strip()


def translate_ge_en(sentence, model, text_vec_layer_en, max_length=MAX_LENGTH):
    """
    Translates a given sentence from German to English using a pre-trained model.

    Parameters:
        sentence (str): The sentence to be translated.
        model (tf.keras.Model, optional): The pre-trained translation model.
        text_vec_layer_en (tf.keras.layers.TextVectorization, optional): The text vectorization layer for English.
        max_length (int, optional): The maximum length of the translated sentence. Defaults to MAX_LENGTH constant.

    Returns:
        str: The translated sentence.
    """
    translation = ""
    punctuation = ""

    # extracting punctuation from the sentence
    if sentence[-1] in ['.', '?', '!']:
        punctuation = sentence[-1]
        sentence = sentence[:-1]

    for word_idx in range(max_length):
        X = np.array([sentence])  # encoder input
        X_dec = np.array(["startofseq " + translation])  # decoder input
        y_proba = model.predict((X, X_dec))[0, word_idx]  # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = text_vec_layer_en.get_vocabulary()[predicted_word_id]

        if predicted_word == "endofseq":
            break

        translation += " " + predicted_word

    translation += punctuation

    return translation.strip()


def eng_to_ger(model, text_vec_layer_de, eng_text, ger_translation):
    """
    Retrieves the text from the `eng_text` widget and checks if it is empty.
    If the text is not empty, it calls the `translate_en_ge` function to perform the translation.
    The translated text is then inserted into the `ger_translation` widget.

    Parameters:
        model (tf.keras.Model): The translation model to use for translation.
        text_vec_layer_de (tf.keras.layers.TextVectorization): The text vectorization layer for German text.
        eng_text (tk.Text): The widget containing the English text to be translated.
        ger_translation (tk.Text): The widget where the translated German text will be inserted.
    """
    text = eng_text.get("1.0", "end-1c")

    if text == "":
        messagebox.showwarning("Warning", "Please, enter the text to translate")
    else:
        translation = translate_en_ge(text, model, text_vec_layer_de)
        ger_translation.delete("1.0", tk.END)
        ger_translation.insert(tk.END, translation)


def ger_to_eng(model, text_vec_layer_en, ger_text, eng_translation):
    """
    Retrieves the text from the `ger_text` widget and checks if it is empty.
    If the text is not empty, it calls the `translate_ge_en` function to perform the translation.
    The translated text is then inserted into the `eng_translation` widget.

    Parameters:
        model (tf.keras.Model): The translation model to use for translation.
        text_vec_layer_en (tf.keras.layers.TextVectorization): The text vectorization layer for English text.
        ger_text (tk.Text): The widget containing the German text to be translated.
        eng_translation (tk.Text): The widget where the translated English text will be inserted.
    """
    text = ger_text.get("1.0", "end-1c")

    if text == "":
        messagebox.showwarning("Warning", "Please, enter the text to translate")
    else:
        translation = translate_ge_en(text, model, text_vec_layer_en)
        eng_translation.delete("1.0", tk.END)
        eng_translation.insert(tk.END, translation)


def instantiate_window():
    """
    Creates and configures the main window for the Neural Machine Translator application.

    Returns:
        root (tk.Tk): The root window object.
    """
    root = tk.Tk()
    root.title("Neural Machine Translator")
    root.iconbitmap(os.path.join(os.getcwd(), "assets/ger_eng.ico"))
    root.option_add("*Button.Font", "Courier 10")

    root.geometry("700x200")

    tk.Label(root, text="English Text:", font=("Courier", 10)).grid(row=0, column=0)
    eng_text = tk.Text(root, height=3, width=40) 
    eng_text.grid(row=1, column=0, padx=10, pady=5)  

    btn_eng_to_ger = tk.Button(root, text="Translate English to German", 
                               command=lambda: eng_to_ger(model_en_ge, text_vec_layer_de, eng_text, ger_translation))
    btn_eng_to_ger.grid(row=2, column=0, padx=10, pady=5) 

    ger_translation = tk.Text(root, height=3, width=40) 
    ger_translation.grid(row=3, column=0, padx=10, pady=5)  

    tk.Label(root, text="German Text:", font=("Courier", 12)).grid(row=0, column=1)
    ger_text = tk.Text(root, height=3, width=40)  
    ger_text.grid(row=1, column=1, padx=10, pady=5)  

    btn_ger_to_eng = tk.Button(root, text="Translate German to English", 
                               command=lambda: ger_to_eng(model_ge_en, text_vec_layer_en, ger_text, eng_translation))
    btn_ger_to_eng.grid(row=2, column=1, padx=10, pady=5)  

    eng_translation = tk.Text(root, height=3, width=40) 
    eng_translation.grid(row=3, column=1, padx=10, pady=5)

    return root


if __name__ == "__main__":
    text_vec_layer_de = create_text_layer_de()
    text_vec_layer_en = create_text_layer_en()

    model_en_ge=create_model_en_ge()
    model_ge_en=create_model_ge_en()

    root = instantiate_window()
    root.mainloop()
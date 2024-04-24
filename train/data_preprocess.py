"""
You should run this script to initialize the vocabularies for the English and German sentences.
N.B. The vocabularies are saved in the models/vocabularies directory as vocab_en.txt and vocab_de.txt. So, feel free to use them.
"""
import os

import tensorflow as tf
import numpy as np

# constants
VOCAB_SIZE = 3000
MAX_LENGTH = 50  # maximum length of a sentence in the dataset
EPOCHS = 10


def load_data():
    """
    Load the English-German translation dataset and returns the English and German sentences as lists.

    Returns:
        tuple(list, list): A tuple containing the English and German sentences as lists (english_sentences, german_sentences).
    """
    translations = []
    data_directory = os.path.join(os.getcwd(), "data/deu.txt")

    with open(data_directory, 'r', encoding="utf8") as file:
        for line in file:
            parts = line.split('\t')
            english = parts[0]  # the English sentence is before the first tab
            german = parts[1]   # the German sentence is after the first tab and before the second
            translations.append((english, german))
    
    np.random.shuffle(translations)

    sentences_en, sentences_de = zip(*translations)

    return (sentences_en, sentences_de)


def initialize_vocabularies_en_de(sentences_en, sentences_de, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    """
    Initialize the text vectorization layers for the English and German sentences, 
    which can be used to translate from English to German.

    Args:
        sentences_en (list): A list of English sentences.
        sentences_de (list): A list of German sentences.
        vocab_size (int): The size of the vocabulary.
        max_length (int): The maximum length of the sentences.
    """
    text_vec_layer_en = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)
    text_vec_layer_de = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)

    text_vec_layer_en.adapt(sentences_en)
    text_vec_layer_de.adapt([f"startofseq {s} endofseq" for s in sentences_de])

    vocab_en = text_vec_layer_en.get_vocabulary()
    vocab_de = text_vec_layer_de.get_vocabulary()

    # save the states of the text vectorization layers
    with open(os.path.join(os.getcwd(), "train/vocabularies_en_de/vocab_en.txt"), 'w', encoding="utf8") as f:
        for item in vocab_en:
            f.write("%s\n" % item)

    with open(os.path.join(os.getcwd(), "train/vocabularies_en_de/vocab_de.txt"), 'w', encoding="utf8") as f:
        for item in vocab_de:
            f.write("%s\n" % item)


def initialize_vocabularies_de_en(sentences_en, sentences_de, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH):
    """
    Initialize the text vectorization layers for the English and German sentences, 
    which can be used to translate from German to English.

    Args:
        sentences_en (list): A list of English sentences.
        sentences_de (list): A list of German sentences.
        vocab_size (int): The size of the vocabulary.
        max_length (int): The maximum length of the sentences.
    """
    text_vec_layer_en = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)
    text_vec_layer_de = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)

    text_vec_layer_de.adapt(sentences_de)
    text_vec_layer_en.adapt([f"startofseq {s} endofseq" for s in sentences_en])

    vocab_en = text_vec_layer_en.get_vocabulary()
    vocab_de = text_vec_layer_de.get_vocabulary()

    # save the states of the text vectorization layers
    with open(os.path.join(os.getcwd(), "train/vocabularies_de_en/vocab_en.txt"), 'w', encoding="utf8") as f:
        for item in vocab_en:
            f.write("%s\n" % item)

    with open(os.path.join(os.getcwd(), "train/vocabularies_de_en/vocab_de.txt"), 'w', encoding="utf8") as f:
        for item in vocab_de:
            f.write("%s\n" % item)


if __name__ == "__main__":
    sentences_en, sentences_de = load_data()
    initialize_vocabularies_en_de(sentences_en, sentences_de)
    initialize_vocabularies_de_en(sentences_en, sentences_de)
    print("Vocabularies initialized successfully!")
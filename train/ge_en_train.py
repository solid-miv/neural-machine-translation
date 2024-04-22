"""
Run this script to train the German to English translation model.
N.B. You can use the already trained model's weights in the models/german-to-english/de_en directory.
"""
import os

import tensorflow as tf

from data_preprocess import load_data

# constants
VOCAB_SIZE = 3000
MAX_LENGTH = 50  # maximum length of a sentence in the dataset
EPOCHS = 10


def split_data_english(sentences_en, sentences_de):
    """
    Split the English and German sentences into training and validation sets.

    Args:
        sentences_en (list): A list of English sentences.
        sentences_de (list): A list of German sentences.
    
    Returns:
        tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor): A tuple containing the training and validation data.
    """
    # load the text vectorization layer for German
    text_vec_layer_en = tf.keras.layers.TextVectorization(VOCAB_SIZE, output_sequence_length=MAX_LENGTH)

    with open(os.path.join(os.getcwd(), "train/vocabularies/vocab_en.txt"), 'r') as f:
        vocabulary_de = [line.strip() for line in f]

    text_vec_layer_en.set_vocabulary(vocabulary_de)

    X_train = tf.constant(sentences_de[:100_000])
    X_valid = tf.constant(sentences_de[100_000:])

    X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_en[:100_000]])
    X_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_en[100_000:]])

    Y_train = text_vec_layer_en([f"{s} endofseq" for s in sentences_en[:100_000]])
    Y_valid = text_vec_layer_en([f"{s} endofseq" for s in sentences_en[100_000:]])

    return (X_train, X_valid, X_train_dec, X_valid_dec, Y_train, Y_valid)
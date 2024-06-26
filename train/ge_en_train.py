"""
Run this script as a module to train the German to English translation model with a command: "python -m train.ge_en_train".
N.B. You can use the already trained model's weights in the models/german-to-english/de_en directory.
"""
import os

import tensorflow as tf

from .data_preprocess import load_data
from .en_ge_train import compile_model, train_model

# constants
VOCAB_SIZE = 3000
MAX_LENGTH = 50  # maximum length of a sentence in the dataset
EPOCHS = 10


def split_data_ge_en(sentences_en, sentences_de):
    """
    Split the English and German sentences into training and validation sets.

    Parameters:
        sentences_en (list): A list of English sentences.
        sentences_de (list): A list of German sentences.
    
    Returns:
        tuple(tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor): A tuple containing the training and validation data.
    """
    # load the text vectorization layer for German
    text_vec_layer_en = tf.keras.layers.TextVectorization(VOCAB_SIZE, output_sequence_length=MAX_LENGTH)

    with open(os.path.join(os.getcwd(), "train/vocabularies_de_en/vocab_en.txt"), 'r') as f:
        vocabulary_en = [line.strip() for line in f]

    text_vec_layer_en.set_vocabulary(vocabulary_en)

    X_train = tf.constant(sentences_de[:100_000])
    X_valid = tf.constant(sentences_de[100_000:])

    X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_en[:100_000]])
    X_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_en[100_000:]])

    Y_train = text_vec_layer_en([f"{s} endofseq" for s in sentences_en[:100_000]])
    Y_valid = text_vec_layer_en([f"{s} endofseq" for s in sentences_en[100_000:]])

    return (X_train, X_valid, X_train_dec, X_valid_dec, Y_train, Y_valid)


def create_architecture_ge_en(embed_size=128, vocab_size=VOCAB_SIZE, 
                        max_length=MAX_LENGTH, N = 2, num_heads=8, dropout_rate=0.1, n_units=128):
    """
    Create the architecture for the German to English translation transformer model.

    Parameters:
        embed_size (int): The embedding size.
        vocab_size (int): The size of the vocabulary.
        max_length (int): The maximum length of the sentences.
        N (int): The number of blocks.
        num_heads (int): The number of attention heads.
        dropout_rate (float): The dropout rate.
        n_units (int): The number of units for the first Dense layer in each Feed Forward block.
    
    Returns:
        tf.keras.Model: The German to English translation transformer model.
    """
    text_vec_layer_de = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)
    text_vec_layer_en = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)

    # Load the vocabulary from the .txt file
    with open(os.path.join(os.getcwd(), "train/vocabularies_de_en/vocab_de.txt"), 'r') as f:
        vocabulary_de = [line.strip() for line in f]
    text_vec_layer_de.set_vocabulary(vocabulary_de)

    with open(os.path.join(os.getcwd(), "train/vocabularies_de_en/vocab_en.txt"), 'r') as f:
        vocabulary_en = [line.strip() for line in f]
    text_vec_layer_en.set_vocabulary(vocabulary_en)
    
    encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
    decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

    encoder_input_ids = text_vec_layer_de(encoder_inputs)
    decoder_input_ids = text_vec_layer_en(decoder_inputs)
    encoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
    decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
    encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
    decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

    pos_embed_layer = tf.keras.layers.Embedding(max_length, embed_size)
    batch_max_len_enc = tf.shape(encoder_embeddings)[1]
    encoder_in = encoder_embeddings + pos_embed_layer(tf.range(batch_max_len_enc))
    batch_max_len_dec = tf.shape(decoder_embeddings)[1]
    decoder_in = decoder_embeddings + pos_embed_layer(tf.range(batch_max_len_dec))

    encoder_pad_mask = tf.math.not_equal(encoder_input_ids, 0)[:, tf.newaxis]
    Z = encoder_in
    for _ in range(N):
        skip = Z
        attn_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_size, dropout=dropout_rate)
        Z = attn_layer(Z, value=Z, attention_mask=encoder_pad_mask)
        Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))
        skip = Z
        Z = tf.keras.layers.Dense(n_units, activation="relu")(Z)
        Z = tf.keras.layers.Dense(embed_size)(Z)
        Z = tf.keras.layers.Dropout(dropout_rate)(Z)
        Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))

    decoder_pad_mask = tf.math.not_equal(decoder_input_ids, 0)[:, tf.newaxis]
    causal_mask = tf.linalg.band_part(  # creates a lower triangular matrix
        tf.ones((batch_max_len_dec, batch_max_len_dec), tf.bool), -1, 0)
    
    encoder_outputs = Z  # let's save the encoder's final outputs
    Z = decoder_in  # the decoder starts with its own inputs
    for _ in range(N):
        skip = Z
        attn_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_size, dropout=dropout_rate)
        Z = attn_layer(Z, value=Z, attention_mask=causal_mask & decoder_pad_mask)
        Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))
        skip = Z
        attn_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_size, dropout=dropout_rate)
        Z = attn_layer(Z, value=encoder_outputs, attention_mask=encoder_pad_mask)
        Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))
        skip = Z
        Z = tf.keras.layers.Dense(n_units, activation="relu")(Z)
        Z = tf.keras.layers.Dense(embed_size)(Z)
        Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))

    Y_proba = tf.keras.layers.Dense(vocab_size, activation="softmax")(Z)

    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba])

    return model


def save_weights_ge_en(model, name):
    """
    Save the German-to-English model's weights.

    Parameters:
        model (tf.keras.Model): The model which weights to save.
        name (str): The name of the file to save the weights to.
    """
    model.save_weights(os.path.join(os.getcwd(), f"models/german-to-english/de_en/{name}"))


if __name__ == "__main__":
    sentences_en, sentences_de = load_data()

    X_train, X_valid, X_train_dec, X_valid_dec, Y_train, Y_valid = split_data_ge_en(sentences_en, sentences_de)

    model = create_architecture_ge_en()

    model = compile_model(model)

    history = train_model(model, X_train, X_train_dec, Y_train, X_valid, X_valid_dec, Y_valid)

    save_weights_ge_en(model, "ge_en")

    print("Model has been trained and saved successfully!")
import os

import tensorflow as tf
import numpy as np

from layers_classes import ShapeLayer, RangeLayer, NotEqualLayer, ExpandDimsLayer, OnesLayer, BandPartLayer    

def load_data():
    translations = []
    data_directory = os.path.join(os.getcwd(), "data/deu.txt")

    with open(data_directory, 'r', encoding="utf8") as file:
        for line in file:
            parts = line.split('\t')
            english = parts[0]  # the English sentence is before the first tab
            german = parts[1]   # the German sentence is after the first tab and before the second
            translations.append((german, english))
    
    np.random.shuffle(translations)

    sentences_de, sentences_en = zip(*translations)

    return (sentences_de, sentences_en)


def initialize_vocabularies(sentences_de, sentences_en, vocab_size=3000, max_length=50):
    text_vec_layer_de = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)
    text_vec_layer_en = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)

    text_vec_layer_de.adapt(sentences_de)
    text_vec_layer_en.adapt([f"startofseq {s} endofseq" for s in sentences_en])

    vocab_de = text_vec_layer_de.get_vocabulary()
    vocab_en = text_vec_layer_en.get_vocabulary()

    # save the states of the text vectorization layers
    with open(os.path.join(os.getcwd(), "models/german-to-english/vocabularies/vocab_de.txt"), 'w', encoding="utf8") as f:
        for item in vocab_de:
            f.write("%s\n" % item)

    with open(os.path.join(os.getcwd(), "models/german-to-english/vocabularies/vocab_en.txt"), 'w', encoding="utf8") as f:
        for item in vocab_en:
            f.write("%s\n" % item)

    return (text_vec_layer_de, text_vec_layer_en)
    
def split_data(sentences_de, sentences_en, text_vec_layer_en):
    X_train = tf.constant(sentences_de[:100_000])
    X_valid = tf.constant(sentences_de[100_000:])

    X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_en[:100_000]])
    X_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_en[100_000:]])

    Y_train = text_vec_layer_en([f"{s} endofseq" for s in sentences_en[:100_000]])
    Y_valid = text_vec_layer_en([f"{s} endofseq" for s in sentences_en[100_000:]])

    return (X_train, X_valid, X_train_dec, X_valid_dec, Y_train, Y_valid)


def create_architecture(text_vec_layer_de, text_vec_layer_en, embed_size=128, vocab_size=3000, 
                        max_length=50, N = 2, num_heads=8, dropout_rate=0.1, n_units=128):
    encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
    decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

    encoder_input_ids = text_vec_layer_de(encoder_inputs)
    decoder_input_ids = text_vec_layer_en(decoder_inputs)

    encoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size,
                                                        mask_zero=True)
    decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size,
                                                        mask_zero=True)
    encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
    decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

    pos_embed_layer = tf.keras.layers.Embedding(max_length, embed_size)
    batch_max_len_enc = ShapeLayer()(encoder_embeddings)
    encoder_in = encoder_embeddings + pos_embed_layer(RangeLayer()(batch_max_len_enc))
    batch_max_len_dec = ShapeLayer()(decoder_embeddings)
    decoder_in = decoder_embeddings + pos_embed_layer(RangeLayer()(batch_max_len_dec))

    N = 2  # number of blocks
    num_heads = 8 # number of attention heads
    dropout_rate = 0.1
    n_units = 128  # for the first Dense layer in each Feed Forward block
    encoder_pad_mask = ExpandDimsLayer()(NotEqualLayer()(encoder_input_ids))
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

    decoder_pad_mask = ExpandDimsLayer()(NotEqualLayer()(decoder_input_ids))
    causal_mask = BandPartLayer()(OnesLayer()(batch_max_len_dec))
    
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
    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
                        outputs=[Y_proba])

    return model


def compile_model(model):
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

    return model


def train_model(model, X_train, X_train_dec, Y_train, X_valid, X_valid_dec, Y_valid, epochs=10):
    history = model.fit([X_train, X_train_dec], Y_train, epochs=epochs, validation_data=([X_valid, X_valid_dec], Y_valid))

    return history


def save_model(model, name):
    model.save(os.path.join(os.getcwd(), f"models/german-to-english/{name}.keras"))


if __name__ == "__main__":
    sentences_de, sentences_en = load_data()

    text_vec_layer_de, text_vec_layer_en = initialize_vocabularies(sentences_de, sentences_en)

    X_train, X_valid, X_train_dec, X_valid_dec, Y_train, Y_valid = split_data(sentences_de, sentences_en, text_vec_layer_en)

    model = create_architecture(text_vec_layer_de, text_vec_layer_en)

    model = compile_model(model)

    history = train_model(model, X_train, X_train_dec, Y_train, X_valid, X_valid_dec, Y_valid)

    save_model(model, 'ge-to-en-model')
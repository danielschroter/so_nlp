import pandas as pd

# example of a model defined with the sequential api
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Masking, concatenate
from losses.weighted_cross_entropy import WeightedBinaryCrossEntropy

# Model inputs: sequence of n x m word embeddings


def create_model(lstm_layer_size=256, embedding_dim=100, output_dim=100, mask_value=0.0, num_mid_dense=0, lstm_dropout=0):
    """
    creates and returns a simple LSTM model
    :param embedding_dim: size of the input embeddings
    :return: model
    """
    x_title = Input(shape=(None, embedding_dim))
    x_body = Input(shape=(None, embedding_dim))
    
    mask_title = Masking(mask_value=mask_value)(x_title)
    mask_body = Masking(mask_value=mask_value)(x_body)
    
    lstm_title = LSTM(lstm_layer_size, dropout=lstm_dropout)(mask_title)
    lstm_body = LSTM(lstm_layer_size, dropout=lstm_dropout)(mask_body)
    
    concat = concatenate([lstm_title, lstm_body])

    next = concat
    for i in range(num_mid_dense):
        next = Dense(128, activation="relu")(next)
    
    output = Dense(output_dim, activation="sigmoid")(next)
    
    model = Model(inputs=[x_title, x_body], outputs=[output])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
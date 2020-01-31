import pandas as pd

# example of a model defined with the sequential api
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Masking, concatenate
from losses.weighted_cross_entropy import WeightedBinaryCrossEntropy

# Model inputs: sequence of n x m word embeddings


def create_model(lstm_layer_size=256, embedding_dim=128, output_dim=100, mask_value=None):
    """
    creates and returns a simple LSTM model
    :param embedding_dim: size of the input embeddings
    :return: model
    """
    x_title = Input(shape=(None, embedding_dim))
    x_body = Input(shape=(None, embedding_dim))
    
    mask_title = Masking(mask_value=mask_value)(x_title)
    mask_body = Masking(mask_value=mask_value)(x_body)
    
    lstm_title = LSTM(lstm_layer_size)(mask_title)
    lstm_body = LSTM(lstm_layer_size)(mask_body)
    
    concat = concatenate([lstm_title, lstm_body])
    
    output = Dense(output_dim, activation="sigmoid")(concat)
    
    model = Model(inputs=[x_title, x_body], outputs=[output])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
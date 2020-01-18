import pandas as pd

# example of a model defined with the sequential api
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Model inputs: sequence of n x m word embeddings


def create_model(lstm_layer_size=32, embedding_dim=128, output_dim=100):
    """
    creates and returns a simple LSTM model
    :param embedding_dim: size of the input embeddings
    :return: model
    """
    model = Sequential()
    model.add(LSTM(lstm_layer_size, input_shape=(None, embedding_dim)))
    model.add(Dense(output_dim, activation="sigmoid"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


if __name__ == "__main__":
    create_model()
    print("successfully created model")
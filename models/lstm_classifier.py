import pandas as pd

# example of a model defined with the sequential api
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Model inputs: sequence of n x m word embeddings


def create_model(embedding_dim=128):
    model = Sequential()
    model.add(LSTM(32, input_shape=(None, embedding_dim)))
    model.add(Dense(32, activation="sigmoid"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


if __name__ == "__main__":
    create_model()
    print("successfully created model")
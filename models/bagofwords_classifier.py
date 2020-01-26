from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Flatten, Activation




def create_model(input_layer_size=256, vocab_size=1000, output_dim=100):
    """
    creates and returns a simple LSTM model
    :param embedding_dim: size of the input embeddings
    :return: model
    """
    model = Sequential()
    model.add(Dense(input_layer_size, input_shape=(vocab_size,)))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('sigmoid'))
    return model


if __name__ == "__main__":
    create_model()
    print("successfully created model")

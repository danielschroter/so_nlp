from data_prep_helpers import *
from evaluation import *
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Flatten, Activation
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.text import text
from sklearn.preprocessing import MultiLabelBinarizer


sample_size = 100000
n_top_labels=100
normalize_embeddings = False
learning_rate = 0.0000000001

print("Loading Data")
data_path = 'C:\\Users\\dschr\\OneDrive\\Dokumente\\Studium\\TUM\\Kurse\\Informatik\\Semester 5\\ADNLP\\data\\pythonquestions\\'
total_data = load_data(data_path)
data = total_data.sample(sample_size)
print("Data Loaded")

print("Preparing Data")
print(data.shape)
data = data[data["tags"].apply(lambda tags: all([isinstance(t, str) for t in tags]))]
print(data.shape)
data = reduce_number_of_tags(data, 100)


train_size = int(len(data) * .8)
train_posts = data['Body_q'][:train_size]
train_tags = data['tags'][:train_size]
test_posts = data['Body_q'][train_size:]
test_tags = data['tags'][train_size:]


vocab_size = 3000
tokenize = text.Tokenizer(num_words=vocab_size)
tokenize.fit_on_texts(train_posts)

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = MultiLabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)
print("Data prepared")

print("Builduing and Training Model")
n_col = y_train.shape[1]
opt = SGD(lr=learning_rate, momentum=0.9)
model = Sequential()
model.add(Dense(256, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dense(n_col))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=["accuracy"])


history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)


print("Evaluation Model")
n_predictions = 300

predictions = model.predict(x_test[:n_predictions])

l_pred = encoder.inverse_transform(binarize_model_output(predictions, threshold=0.1))
l_true = encoder.inverse_transform(y_test[:n_predictions])
raw_texts = test_posts[:n_predictions]

for pred, act, txt, i in zip(l_pred, l_true, raw_texts, range(2)):
    print(f"TRUE: {act}\nPREDICTION: {pred}\n")
    print(txt)

l_pred_binary = binarize_model_output(predictions, 0.2)
l_true_binary = y_test[:n_predictions]
output_evaluation(model, sample_size, None, n_top_labels, l_true_binary, l_pred_binary, normalize_embeddings, learning_rate, vocab_size)

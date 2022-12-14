from typing import Callable, Dict, List, Tuple

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import time
import pickle
import itertools


def grid_search_es(X, y, create_model: Callable, search_params: Dict[str, List], max_epochs=100) -> List[Tuple]:
    """
    conducts grid search on all combinations of parameters supplied in search_params. Each parameter combination will be
    trained on 3 splits and the best validation losses of each of the 3 runs are averaged to score the parameter
    combination. Each training run is conducted using EarlyStopping until a maximum of max_epochs epochs.

    Results are automatically pickled in a timestamped file.

    :param create_model: parametrized function that returns a model
    :param search_params: dictionary of "str": [int, int, ...] entries that list possible values for each required
                          parameter of create_model
    :param max_epochs: maximum number of epochs for each training
    :return: List of n_combinations 3-tuples containing: (paremters_dict, training_histories, avg_best_val_loss)
    """
    all_hists = []

    split_states = [42, 2020]

    # print(list(search_params.values()))
    # print(list(itertools.product(*list(search_params.values()))))

    for params_l in itertools.product(*list(search_params.values())):
        params = {k: v for k, v in zip(search_params.keys(), params_l)}
        print(params)
        t_start = time.time()
        # we use EarlyStopping with patience 10 because our models tend to only start improving after about 5-10 epochs
        # of training
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10),
        ]

        hists = []
        val_loss = 0.0
        for rs in split_states:
            # if X is a list of title and body training data, we need to take some extra steps to splitting it into
            # train and test sets
            if len(X) == 2:
                # list(zip(*X)) converts the X = [X_title, X_body] to a list of 2-tuples [(X_title_i, X_body_i), ...]
                # where each tuple contains a single X_title and X_body element. This is necessary because
                # train_test_split splits a single list.
                X_train_z, X_test_z, y_train, y_test = train_test_split(list(zip(*X)), y, test_size=0.2,
                                                                        random_state=rs)
                # we need to "unzip" the zipped X_train_z and X_test_z because for the fit call, we again need a list of
                # length 2 that contains a list of X_title and X_body in the form X_test = [X_test_title, X_test_body]
                # where X_test_title and X_test_body are lists of embedded title / body tokens
                X_train = list(zip(*X_train_z))
                X_test = list(zip(*X_test_z))
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
            model = create_model(**params)
            hist = model.fit(X_train, y_train, batch_size=128, validation_data=[X_test, y_test], epochs=max_epochs,
                             verbose=0, callbacks=callbacks)
            hists.append(hist.history)
            val_loss += min(hist.history["val_loss"]) / len(split_states)

        epoch_lengths = [len(h["val_loss"]) for h in hists]
        print(
            f"average min val_loss: {val_loss} -- epochs: {epoch_lengths} -- time: {time.time() - t_start:.2f} seconds")

        all_hists.append((params, hists, val_loss))

    with open(f"logs/hists_log_{int(time.time())}.pkl", "wb") as out_file:
        pickle.dump(all_hists, out_file)

    return all_hists

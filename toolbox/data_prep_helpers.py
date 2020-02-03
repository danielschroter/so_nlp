import pandas as pd
import pickle
import joblib
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec, FastText
import numpy as np
import os
import math


def sent_tokenize_text(txt):
    """
    tokenize a single body of text that may consist of multiple sentences. This function will usually be applied to
    single question/answer bodies as data-preparation for training/applying embeddings
    :param txt: string containing some text
    :return: list of preprocessed single sentences
    """
    txt = ' '.join(txt.split())  # replaces all whitespace with single space (including line-breaks)
    sents = sent_tokenize(txt)
    sents = [s[:-1] if s.endswith(".") else s for s in sents]
    sents = [s.replace(",", "") for s in sents]
    return sents


def word_tokenize_sent(s):
    """tokenizes all words in a sentence (after lower-casing them)"""
    return [w.lower() for w in word_tokenize(s)]


def load_fasttext_embeddings(path):
    """
    Loads trained embeddings
    :param path: path to the location of the embeddings
    :return:
    """
    with open(path, "rb") as in_file:
        return pickle.load(in_file)


def tokenize_text(dataframe, textcolumn):
    """
    Helper Function that generates word tokens on a sentence level
    :param dataframe: Data
    :param textcolumn: Columname of dataframe containing the text
    :return: nested list with tokens for each sentence.
    """
    sentences = []
    for i, row in dataframe.iterrows():
        txt = row[textcolumn]
        sents = sent_tokenize_text(txt)
        sentences += sents

    data = []
    print(sentences[:2])
    for s in sentences:
        data.append(word_tokenize_sent(s))
    print(data[0])
    return data


def generate_question_level_tokens(txt):
    """
    Method generates a list where each entry is a list containing the words per question
    :param txt: txt to be tokenized
    :return: a flattened nested list containing the tokens per question
    """
    sents = sent_tokenize_text(txt)
    words = [word_tokenize_sent(s) for s in sents]
    return [item for sublist in words for item in sublist]  # flatten nested list


def reduce_number_of_tags(dataframe, topTags):
    """
    # Reduce_number_of_tags within the dataset, and adjust the dataframe so that no empty tag entries are present
    :param dataframe: Input Data, with 'tag' column,
    :param topTags: Number of tags that should be used further
    :return:
    """
    all_tags = np.array([tag for tags in dataframe["tags"] for tag in tags])
    unique_tags = np.unique(all_tags, return_counts=True)
    top_tags = unique_tags[0][np.argsort(unique_tags[1])[-101:]]  # keep 101, we will remove "python"
    print(f"deleting element {top_tags[-1]} from top_tags")
    top_tags = np.delete(top_tags, -1)

    # remove tags that are not within top_101 and remove "python" tag
    dataframe["tags"] = dataframe["tags"].apply(lambda x: [tag for tag in x if tag in top_tags])
    # remove any tag lists that we potentially made empty by doing this
    dataframe = dataframe[dataframe["tags"].apply(len) > 0]
    return dataframe


def create_Word2Vec_embeddings(dataframe, textcolumn):
    """
    Code Citation: https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
    :param dataframe: dataframe containg text which should be used for word-embedding
    :param textcolumn: string that specifies the column containing training-text
    :return: word_vectors as Word2VecKeyedVectors object( matrix containing similarity values)
    """
    data = tokenize_text(dataframe, textcolumn)
    model = gensim.models.Word2Vec(data, min_count = 1, size = 100,
                                                 window = 5, sg = 1)
    return model.wv


def create_FastText_embeddings(dataframe, textcolumn):
    #data = tokenize_text(dataframe, textcolumn)
    print("Start training of WE")
    data = dataframe["q_all_body_tokenized"].tolist()
    model = FastText(min_count=1, size=100, window=3, sg=1)
    model.build_vocab(sentences=data)
    model.train(sentences= data, total_examples=len(data), epochs=10)
    return model.wv


def load_data(data_path, drop_extra_columns=True, ignore_cache=False, tokenized_field="q_all_body_tokenized",
              content_field="Body_q", include_answers=False):
    """
    loads question answer and tag data into one dataframe
    :param data_path: folder where Questions.csv, Answers.csv, Tags.csv are found. Must end with / (or \\ for Windows?)
    :param drop_extra_columns: whether to drop columns that are currently not relevant to the model (excl. question Id)
    :return: DataFrame where each row is one question with its top answer and a list of tags
    """

    cache_folder = "cache"

    c1_path = f"{cache_folder}/{data_path.split('/')[-2]}_0.pkl"
    pkl_path = f"{cache_folder}/{data_path.split('/')[-2]}.pkl"
    # load cache data pickle if it exists
    if not ignore_cache and os.path.exists(c1_path):
        print("loading data from cached pickle")
        return load_cached_chunked_data(c1_path.replace("_0.pkl", ""))
    print("loading data from csv files")

    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    questions = pd.read_csv(f"{data_path}Questions.csv", encoding="ISO-8859-1")
    tags = pd.read_csv(f"{data_path}Tags.csv", encoding="ISO-8859-1")

    print("grouping tags")
    grouped_tags = tags.groupby("Id").apply(lambda group: group["Tag"].tolist())

    if include_answers:
        print("loading and merging answers")
        answers = pd.read_csv(f"{data_path}Answers.csv", encoding="ISO-8859-1")
        top_answers = answers.groupby("ParentId").apply(lambda group: group.loc[group["Score"].idxmax()])
        df = questions.merge(top_answers, how="inner", left_on="Id", right_index=True, suffixes=("_q", "_a"))
        del answers, top_answers
    else:
        df = questions
        df["Body_q"] = df["Body"]
        df.drop(["Body"], axis=1, inplace=True)

    print("merging data")
    df = df.merge(grouped_tags.rename("tags"), how="left", left_on="Id", right_on="Id", suffixes=("", "_t"))

    del questions, tags, grouped_tags
    if drop_extra_columns:
        print("dropping extra columns")
        if include_answers:
            # df.drop(["Id_a", "OwnerUserId_a", "CreationDate_a", "Score_a"], inplace=True)
            df.drop(["Id_q", "OwnerUserId_q", "CreationDate_q", "Score_q", "Id_a", "OwnerUserId_a", "CreationDate_a",
                 "ParentId", "Score_a"], axis=1, inplace=True)
        else:
            df.drop(["OwnerUserId", "CreationDate", "Score"], axis=1, inplace=True)

    chunk_size = 10000
    num_chunks = math.ceil(df.shape[0] / chunk_size)
    for i in range(num_chunks):
        chunk = df[i*chunk_size: (i+1)*chunk_size]
        print(f"parsing chunk {i+1}/{num_chunks}")
        #print(f"{i}: removing html tags")

        remove_html_tags(chunk, ["Body_q"])
        #print(f"{i}: generating question level tokens")
        chunk[tokenized_field] = chunk[content_field].apply(generate_question_level_tokens)
        chunk = chunk[tokenized_field.apply(len) > 0]

        # cache resulting dataframe as pickle
        #print(f"{i}: caching dataframe")
        # with open(pkl_path, "wb") as out_file:
        chunk_path = pkl_path.replace(".pkl", f"_{i}.pkl")
        joblib.dump(chunk, chunk_path)

    del df
    return load_cached_chunked_data(pkl_path.replace(".pkl", ""))


def load_cached_chunked_data(prefix):
    """

    :param prefix:
    :return:
    """
    i = 0
    df = None
    while True:
        chunk_path = prefix + f"_{i}.pkl"
        if not os.path.exists(chunk_path):
            return df
        chunk = joblib.load(chunk_path)
        #print(chunk.shape)
        #print(chunk.iloc[0])
        if df is None:
            df = chunk
        else:
            df = df.append(chunk)
        i += 1


def remove_html_tags(dataframe, columnnames):
    """
    Removes html-tags from specified columns
    :param dataframe: input dataframe
    :param columnnames: array of columnames in which the html tags should be removed
    :return: modified dataframe
    """
    for name in columnnames:
        dataframe[name] = dataframe[name].apply(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    return dataframe



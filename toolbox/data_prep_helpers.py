import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec


def create_Word2Vec_embeddings(dataframe, textcolumn):
    """
    Code Citation: https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
    :param dataframe: dataframe containg text which should be used for word-embedding
    :param textcolumn: string that specifies the column containing training-text
    :return: word_vectors as Word2VecKeyedVectors object( matrix containing similarity values)
    """
    str = " "
    for i, row in dataframe.iterrows():
        str = str + row[textcolumn]

    f = str.replace("\n", " ")
    data = []
    for i in sent_tokenize(f):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        data.append(temp)

    model = gensim.models.Word2Vec(data, min_count = 1, size = 100,
                                                 window = 5, sg = 1)

    # print(model.most_similar('python'))
    word_vectors = model.wv
    return word_vectors


def load_data(data_path, drop_extra_columns=True):
    """
    loads question answer and tag data into one dataframe
    :param data_path: folder where Questions.csv, Answers.csv, Tags.csv are found. Must end with / (or \\ for Windows?)
    :param drop_extra_columns: whether to drop columns that are currently not relevant to the model (excl. question Id)
    :return: DataFrame where each row is one question with its top answer and a list of tags
    """
    questions = pd.read_csv(f"{data_path}Questions.csv", encoding="ISO-8859-1")
    answers = pd.read_csv(f"{data_path}Answers.csv", encoding="ISO-8859-1")
    tags = pd.read_csv(f"{data_path}Tags.csv", encoding="ISO-8859-1")

    grouped_tags = tags.groupby("Id").apply(lambda df: df["Tag"].tolist())
    top_answers = answers.groupby("ParentId").apply(lambda df: df.loc[df["Score"].idxmax()])

    df = questions.merge(top_answers, how="inner", left_on="Id", right_index=True, suffixes=("_q", "_a"))
    df = df.merge(grouped_tags.rename("tags"), how="left", left_on="Id", right_on="Id", suffixes=("", "_t"))

    if drop_extra_columns:
        df.drop(["Id_q", "OwnerUserId_q", "CreationDate_q", "Score_q", "Id_a", "OwnerUserId_a", "CreationDate_a",
                 "ParentId", "Score_a"], axis=1, inplace=True)
    return df


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

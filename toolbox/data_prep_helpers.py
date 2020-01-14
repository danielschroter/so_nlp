import pandas as pd
from bs4 import BeautifulSoup



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

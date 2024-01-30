import numpy as np
import pandas as pd

from scipy.spatial import distance
from abc import ABC, abstractmethod
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity



class NLP_General(ABC):
    def __init__(self, **kwargs):
        """Build a nlp object from the input data"""
    
    def process_text(self, text):
        pass

class NLP_File(NLP_General):
    """Build a nlp object from a file. Will later distinguish between other types of datasources (ie: webpage, audio, etc)."""

    __name__ = 'NLP_File'

    def __init__(self, file_path, dataframe=None):
        super().__init__()  # Call the constructor of the superclass if needed
        self.file_path = file_path
        self.dataframe = dataframe

    def display_dataframe(self) -> pd.DataFrame:
        if self.dataframe is not None:
            print("DataFrame loaded from file:")
            return self.dataframe
        else:
            print("No DataFrame loaded.")
    
    ## Euclidean Distance
    
    def euclidean_function(self, vectors):
        euclideans = []
        euc=euclidean_distances(vectors[0], vectors[1])
        euclideans.append(euc)

        return euclideans

    
    def convert(self, dist):
        new_df = pd.DataFrame()
        arr = np.array(dist)
        arr = np.concatenate(arr, axis=0)
        arr = np.concatenate(arr, axis=0)
        
        new_df['euclidean'] = arr

        return new_df
    def tfidf(self, string1: str, string2: str):
        """"TF-IDF"""
        ques = []

        x = self.display_dataframe().iloc[:, 1:5]
        x = x.dropna(how = 'any')
        
        for k in range(len(x)):
            for j in [2, 3]:
                ques.append(x.iloc[k, j])
        vect = TfidfVectorizer()
        # Fit the your whole dataset. After all, this'll 
        # produce the vectors which is based on words in corpus/dataset
        vect.fit(ques)
    
        corpus = [string1, string2]
        trans = vect.transform(corpus)
    
        edu = self.euclidean_function(trans)
        edu_score = self.convert(edu)
        # self.cosine(trans)
        # self.manhatten_distance(trans)

        return edu_score


from __future__ import annotations
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

class NLP_General(ABC):
    def __init__(self, **kwargs):
        """Build a nlp object from the input data"""
    
    def process_text(self, text):
        pass

class NLP_File(NLP_General):

    __name__ = 'NLP_File'

    def __init__(self, file_path, dataframe=None):
        super().__init__()  # Call the constructor of the superclass if needed
        self.file_path = file_path
        self.dataframe = dataframe

    def display_dataframe(self):
        if self.dataframe is not None:
            print("DataFrame loaded from file:")
            return self.dataframe
        else:
            print("No DataFrame loaded.")
    

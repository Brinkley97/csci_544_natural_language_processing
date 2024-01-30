import os.path

import numpy as np
import pandas as pd

from natural_language_processing import NLP_File

def build_quora_duplicate_questions(file_path: pd.DataFrame, delim: str, encoder: str) -> NLP_File:
    data_df = pd.read_csv(file_path, delimiter=delim, encoding=encoder)

    return NLP_File(file_path, data_df)

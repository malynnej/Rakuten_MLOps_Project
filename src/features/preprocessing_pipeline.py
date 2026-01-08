###########################################################################################
#
# Main Data Dreprocessing Pipeline
#
# 1. Textual Data Preprocessing
#
# 1.1 Text Cleaning @Rohan
# 1.2 Text Translation @Rohan
# 1.3 Text outliers @Michael
# 1.4 Balancing Textual Data @Johann
#
# 2. Image Data Preprocessing
#
# 2.1 Clean Image Data @Michael
# 2.2 Image Augmentation @Jenny
#
###########################################################################################

# Import libraries
import pandas as pd
import numpy as np
import os
import time
from datetime import timedelta


# Import custom classes for data preprocessing
from src.features.initialize_data import initialization
from src.features.text_cleaning import TextCleaning
from src.features.text_outliers import TransformTextOutliers

# for download of llm model 
import spacy
from spacy.cli import download


# Loading Data Frame
init = initialization(dev_mode=False)
df_train, df_test, paths, params = init.initialize()

# download llm model 
try:
    nlp = spacy.load(params.TextOutlier.llm)
except OSError:
    print("Downloading llm model...")
    download(params.TextOutlier.llm)

def preprocessing_pipeline(df, paths, params, mode):

    ###########################################################################################

    # 1. Textual Data Preprocessing

    ###########################################################################################

    # 1.1 Text Cleaning @Rohan
    if params.ExecFlags.TextCleaningFlag:
        clt = TextCleaning(html_to_text = params.DecodeEncode.html_to_text,
                        words_encoding = params.DecodeEncode.words_encoding)

        df,_, = clt.cleanTxt(df,["text"])

    # 1.2 Text outliers @Michael
    if params.ExecFlags.TransformTextOutliersFlag:
        tto = TransformTextOutliers(column_to_transform="text",
                                    model=params.TextOutlier.llm,
                                    word_count_threshold = params.TextOutlier.word_count_threshold,
                                    sentence_normalization=params.TextOutlier.sentence_normalization,
                                    similarity_threshold=params.TextOutlier.similarity_threshold, 
                                    factor=params.TextOutlier.factor)

        df, _, _ = tto.transform_outliers(df)
    


    init.save_files(df, paths, mode, X_columns=["text"])


# Start execution timer
t_start = time.time()

print(">>> Execute preprocessing pipeline for training data.")
preprocessing_pipeline(df_train, paths, params, mode="train")

print(">>> Execute preprocessing pipeline for test data.")
preprocessing_pipeline(df_test, paths, params, mode="test")

# End execution timer
t_end = time.time()
t_exec = str(timedelta(seconds=t_end-t_start))

print("\nData preprocessing done!")
print(f"Execution time: {t_exec}.")


###########################################################################################

# Results

###########################################################################################

# Expected output:
# .csv-files:
# X_train, X_test, y_train, y_test
#

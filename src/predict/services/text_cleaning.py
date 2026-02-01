import html
import time

import ftfy
from bs4 import BeautifulSoup
from tqdm import tqdm

tqdm.pandas()
###########################################################################################
#
# Main Data Dreprocessing Pipeline
#
# 1. Textual Data Preprocessing
#
# 1.1 Text Cleaning - Converting the text from html to normal text Fixing Some
#   encoded words into normal ones

# 1.2 Text Translation @Rohan
#
###########################################################################################


# Import important libraries

# ----------------------------------
# Data & Numerical Analysis
# ----------------------------------


# ----------------------------------
# Text Processing & NLP
# ----------------------------------


class TextCleaning:
    """
    This class can be used to identify and transform the DataFrame text from
    html to normal text. Moreover, there are findings that some of text words
    consists of encoded words which will be converted into plausible words.

    Parameters :
    ----------------------------
         html_to_text : bool, optional (default=True)
            If True, converts HTML content into plain text.

        words_encoding : bool, optional (default=True)
            If True, fixes encoding issues (mojibake, smart quotes, etc.).

    """

    def __init__(self, html_to_text=True, words_encoding=True):
        self.html_to_text = html_to_text
        self.words_encoding = words_encoding
        self.corrected_words = {}  # Tracks replaced words

        print("TextCleaning object initialized successfully....")

    # -----------------------------------------------------------
    # ---------------     clean text function   -----------------
    # -----------------------------------------------------------

    def cleanTxt(self, df, columns):
        """
        Run the text cleaning pipeline on specific columns.

        Parameter :
        --------------
        df : pandas.DataFrame - input Data frame containing text data columns :
        List of columns to clean

        Return :
        -------------

        df : Cleaned DataFrane

        """

        # Start execution timer
        t_start = time.time()

        df = df.copy()

        if self.html_to_text:
            df = self._html2text(df, columns)

        if self.words_encoding:
            df = self._fix_encoding(df, columns)

        # End execution timer
        t_end = time.time()
        t_exec = t_end - t_start
        print(f"Execution time: {t_exec} seconds.")

        return df, self.corrected_words

    # -----------------------------------------------------------
    # ---------     Convert html to plain text   ----------------
    # -----------------------------------------------------------

    def _html2text(self, df, columns):
        """Vectorized HTML to text conversion - much faster than apply()"""

        for col in columns:
            if col in df.columns:
                # Vectorized HTML unescape
                df[col] = df[col].fillna("").astype(str).apply(html.unescape)

                # BeautifulSoup parsing (unavoidable apply, but without progress
                # bar)
                df[col] = df[col].apply(
                    lambda x: BeautifulSoup(x, "html.parser").get_text(separator="").strip()
                    if x
                    else x
                )
            else:
                print(f" Warning: Column '{col}' not found in DataFrame.")

        return df

    # -----------------------------------------------------------
    # ---------     Fix text encoding issues   ------------------
    # -----------------------------------------------------------

    def _fix_encoding(self, df, columns):
        """Vectorized encoding fix - removed progress bars for speed"""

        def fix_and_track(text):
            text = str(text)
            fixed = ftfy.fix_text(text)
            old_words = text.split()
            fixed_words = fixed.split()

            for o, f in zip(old_words, fixed_words):
                if o != f:
                    self.corrected_words[o] = f
            return fixed

        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(fix_and_track)
            else:
                print(f" Warning: Column '{col}' not found in DataFrame.")

        return df

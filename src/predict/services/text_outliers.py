# General
# Execution time
import time
from string import punctuation

# Text preprocessing
import nltk
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS

# Display progress bars
from tqdm import tqdm

tqdm.pandas()



class TransformTextOutliers:
    """
    This class can be used to identify and analyse outliers in text within a data frame. These outliers can then be transformed to meet the necessary requirements. 
    The following functions are provided:

    1. text_stats_calc(): Calculates descriptive statistics based on word and sentence measures. 
    2. outlier_analysis(): Analyzes and visualizes text outliers regarding the results of the previous function. A threshold can be defined to define text outliers.
    3. tokenize_sents(): Sentence tokenization and optional normalization based on nltk library (sent_tokenize).
    4. cosine_similarity_calc(): Uses TF-IDF vectors to compute cosine similarity between sentences, identifying and condensing redundant text information.
    5. word_frequency(): Analyzes text to identify the frequency of non-stop words and non-punctuation tokens to calculate word frequency for text summarization.
    6. summarizer(): Produces text summarization, retaining the most important sentences based on word frequencies.
    7. transform_outliers(): Applies all previous functions to transform text outliers in a data frame.

    Parameters:
    -----------------------------
    column_to_transform (str): Column name to transform
    model: Pre-trained LLM from spaCy library for summarization of text. It analyzes the syntax and structure of the text to provide sentence boundaries. (default: en_core_web_en)
    word_count_threshold (int): Threshold for outlier detection
    sentence_normalization: Lowercasing sentences. (default: True)
    similarity_threshold: Defines the threshold for sentence similarity. (default: 0.8)
    factor: Factor for calculating the word count threshold based on average word counts and standard deviation. (default: 3)

    Returns:
    -----------------------------
    transformed_dataframe: Data frame with transformed text column.
    stats_init: Dirctionary of statistical text measures pre-transformation.
    stats_post: Dirctionary of statistical text measures post-transformation.

    """

    def __init__(self, column_to_transform="text", model="en_core_web_en", word_count_threshold=250, sentence_normalization=True, similarity_threshold=0.8, factor=3):
        self.column_to_transform = column_to_transform
        self.model = model
        self.nlp = spacy.load(self.model)
        self.word_count_threshold = word_count_threshold
        self.sentence_normalization = sentence_normalization
        self.similarity_threshold = similarity_threshold
        self.factor = factor
        self.log = []

        print("#" * 50, "\n")
        print("TransformOutlier object initialized successfully.")
        print("Current model: ", self.model)
        if self.sentence_normalization:
            print("Sentence normalization is activated.")
        print("Similarity threshold is: ", self.similarity_threshold)
        print(f"Threshold of word counts is {self.factor} * standard deviation")
        print("Defined word count threshold: ", self.word_count_threshold)
        print("\n")
        print("#" * 50)

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt...")
            nltk.download('punkt')

    # Text statistics in data frame
    def text_stats_calc(self,dataframe):
        """
        This function provides a statistical calculation of word and sentence mesures.

        Input:
        -----------------------------
        dataframe: Data frame containing "text" column.

        Returns:
        -----------------------------
        stats: Dictionary containing statistical measures (min, max, mean, std) of word and sentences counts. 
        word_counts: Series of counted words per row.
        sentence_counts: Series of counted sentences per row.

        """
        
        # Copy of data frame
        df = dataframe.copy()

        # Word count
        word_counts = df.apply(lambda row: len(row.split(" ")))

        # Calculate word count statistics
        stats = {"word_count_min": np.min(word_counts),
                "word_count_max": np.max(word_counts),
                "word_count_mean": np.mean(word_counts),
                "word_count_std": np.std(word_counts)}

        # Print word count statistics
        print("\nWord counts:")
        print("Range of word counts: ", stats["word_count_min"], " to ", stats["word_count_max"], " words.")
        print("Average: ", stats["word_count_mean"], " +/- ", stats["word_count_std"])

        # Sentence count
        sentence_counts = df.apply(lambda row: len(self.tokenize_sents(row)))

        # Calculate sentence count statistics
        stats.update({"sentence_count_min": np.min(sentence_counts),
                    "sentence_count_max": np.max(sentence_counts),
                    "sentence_count_mean": np.mean(sentence_counts),
                    "sentence_count_std": np.std(sentence_counts)})

        # Print sentence count statistics
        print("\nSentence counts:")
        print("Range of sentence counts: ", stats["sentence_count_min"], " to ", stats["sentence_count_max"], " sentences.")
        print("Average: ", stats["sentence_count_mean"], " +/- ", stats["sentence_count_std"])

        return stats, word_counts, sentence_counts


    # Analyze outliers
    def outlier_analysis(self, dataframe, factor, display_graph=False):
        """
        This function provides a analysis of outliers for word and sentence.

        Input:
        -----------------------------
        dataframe: Data frame containing "text" column.
        factor: Factor for calculating the word count threshold based on average word counts and standard deviation. (default: 3)
        display_graph: Activates graphical representations of outlier analysis (boxenplots of word and sentence counts). (default: False)

        Returns:
        -----------------------------
        violated_obeservation: Data frame containing all observations with text outliers.
        stats: Dictionary containing statistical measures (min, max, mean, std) of word and sentences counts. 

        """

        # Copy of data frame
        df = dataframe.copy()

        # Calculate test statistics
        stats, word_counts, sentence_counts = self.text_stats_calc(df[self.column_to_transform])

        df["word_counts"] = word_counts
        df["sentence_counts"] = sentence_counts

        # Calcuate thresholds
        word_count_threshold = stats["word_count_mean"] + factor * stats["word_count_std"]
        print("Outlier threshold: ", stats["word_count_mean"], " + ", factor, " * ", stats["word_count_std"], " = ",  word_count_threshold)
        print("Defined word count threshold: ", self.word_count_threshold)

        print("\nDescriptive statistics of word counts:")
        print(df["word_counts"].describe())


        # Get data frame containing outliers according to threshold.
        violated_obeservation = df["word_counts"] >= self.word_count_threshold

        return violated_obeservation, stats
        
    # Tokenize sentences
    def tokenize_sents(self, sentences):
        """
        This function provides tokenized sentences with nltk library using the method sent_tokenize().
        The sentences can be normalized (lowercased) optionally.
        The main purpose of this function is to provide customized tokenization for calculationg cosine similarity.

        Input:
        -----------------------------
        sentences: Text strings containing multiple sentences.

        Returns:
        -----------------------------
        tokenized_sentences: List of tokenized sentences.

        """

        # Tokenize sentences
        sentences = sent_tokenize(sentences)

        # Sentence normailzation
        if self.sentence_normalization:
            tokenized_sentences = [sentence.lower() for sentence in sentences]
        else:
            tokenized_sentences = sentences

        return tokenized_sentences

    # Calculate cosine similarity between sentences
    def cosine_similarity_calc(self, sentences, display_similarity=False):
        """
        This function calculates cosine similarity of tokenized sentences.
        Senetences with cosine similarity according to defined threshold are eliminated.  

        Input:
        -----------------------------
        sentences: Text strings containing multiple sentences.
        display_similarity: Activates heatmap with cosine similarites. (default: False) 
        -> (Caution: This fuctionallity is only for single string analysis and NOT for comlete data frame due to high computational costs)

        Returns:
        -----------------------------
        cleaned_sentences: String with cleaned sentences.

        """

        # Tokenize sentences
        tokenized_sentences = self.tokenize_sents(sentences)

        # Vectorize tokenized sentences
        vectorizer = TfidfVectorizer()
        tfidf_mat = vectorizer.fit_transform(tokenized_sentences)

        # Calculate cosine similarity
        cosine_sim_matrix = cosine_similarity(tfidf_mat)

        # Identify similar sentences
        indices_to_remove = []

        for i in range(cosine_sim_matrix.shape[0]):
            for j in range(i + 1, cosine_sim_matrix.shape[0]):
                if cosine_sim_matrix[i, j] > self.similarity_threshold:
                    self.log.append(f"\nSentence {i} and Sentence {j} are similar:")
                    self.log.append(tokenized_sentences[i])
                    self.log.append(tokenized_sentences[j])

                    indices_to_remove.append(j)

        self.log.append(f"\nRemoved: {len(indices_to_remove)} sentences.")

        # Remove redundant sentences and join them to a string
        cleaned_sentences = [tokenized_sentences[i] for i in range(len(tokenized_sentences)) if i not in indices_to_remove]
        cleaned_sentences = " ".join(cleaned_sentences)

        return cleaned_sentences

    # Calculate word frequency
    def word_frequency(self, doc):
        """
        This function calculates the word frequency of significant words within a doc object.
        Stop words are not taken into account.  

        Input:
        -----------------------------
        doc: document object (spaCy)

        Returns:
        -----------------------------
        word_frequencies: Dictionary of unique, non-stop-word, and non-punctuation token according to number of occurances in document.

        """

        # Calculate word frequencies
        word_frequencies = {}
        for token in doc:
            if token.text not in STOP_WORDS and token.text not in punctuation:
                if token.text not in word_frequencies:
                    word_frequencies[token.text] = 1
                else:
                    word_frequencies[token.text] += 1

        return word_frequencies

    # Summarize text
    def summarizer(self, text):
        """
        This function summarizes given text, retaining the most important sentences based on word frequencies.
        It sorts the sentences within the document object according to the sum of the frequencies of tokens they contain.
        The three sentences with the highest scores, indicating they consist of the most frequently occurring words, are retained to form the summary.

        Input:
        -----------------------------
        text: Textual data.

        Returns:
        -----------------------------
        summary: Summarized text.

        """

        # Create doc object
        doc = self.nlp(text)

        # Calculate word frequencies
        word_frequencies = self.word_frequency(doc)

        # Sort sentences according to the sum of the frequencies of tokens
        sorted_sentences = sorted(doc.sents, key=lambda sent: sum(word_frequencies[token.text] for token in sent if token.text in word_frequencies), reverse=True)

        # Join 3 sentences with highest scores
        summary = " ".join(sent.text for sent in sorted_sentences[:3])

        self.log.append("\nCompare original and summarized texts:")
        self.log.append(text)
        self.log.append(summary)

        return summary
    
    def transform_outliers(self, dataframe):
        """
        This function applies statistical text analysis, cleaning of redundant sentences and summarization according threshold of word counts.

        Input:
        -----------------------------
        dataframe: Data frame containing "text" column.

        Returns:
        -----------------------------
        transformed_dataframe: Data frame with transformed text column.
        stats_init: Dictionary of statistical text measures pre-transformation.
        stats_post: Dictionary of statistical text measures post-transformation.

        """

        # Start execution timer
        t_start = time.time()

        # Copy data frame
        df = dataframe.copy()

        # define threshold for text length
        print("\n>>> Initial analysis:")
        violated_observation, stats_init = self.outlier_analysis(df, self.factor)

        self.log.append("#" * 50)
        self.log.append("\nCosine similarity:\n")
        self.log.append("#" * 50)

        # Remove redundant informations
        print("\n>>> Cleaning sentences.")
        if len(violated_observation) > 0:
            df.loc[violated_observation, self.column_to_transform] = df.loc[violated_observation, self.column_to_transform].apply(lambda row: self.cosine_similarity_calc(row))

        self.log.append("#" * 50)
        self.log.append("\nText summary:\n")
        self.log.append("#" * 50)

        # Summarize texts
        print(">>> Starting text summarization.")
        if len(violated_observation) > 0:
            df.loc[violated_observation, self.column_to_transform] = df.loc[violated_observation, self.column_to_transform].apply(lambda row: self.summarizer(row))

        # Calculate text statistics post transformation
        print(">>> Post analysis.")
        stats_post, _, _ = self.text_stats_calc(df[self.column_to_transform])

        transformed_dataframe = df

        # End execution timer
        t_end = time.time()
        t_exec = t_end - t_start

        print("\nOutlier transformation finished successfully!.\n")
        print(f"Execution time: {t_exec:.2f} seconds.")

        return transformed_dataframe, stats_init, stats_post

    




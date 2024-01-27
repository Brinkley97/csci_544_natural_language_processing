"""
Detravious Jamari Brinkley
HW1
CSCI-544: Applied Natural Language Processing

python version: 3.11.4

"""
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

import sklearn
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset = "../datasets/amazon_reviews_us_Office_Products_v1_00.tsv"
amazon_reviews_copy_df = pd.read_csv(dataset, sep='\t', on_bad_lines='skip', low_memory=False)

reviews_ratings_df = amazon_reviews_copy_df.loc[0:, ['star_rating', 'review_body']]
reviews_ratings_df.reset_index(drop=True)

reviews_ratings_df['review_body'].astype(str)
reviews_ratings_df

average_length_before_cleaning = reviews_ratings_df['review_body'][reviews_ratings_df['review_body'].apply(type) == str].str.len().mean()
print()
print("Average length of the reviews in terms of character length BEFORE cleaning", average_length_before_cleaning)
print()


def generate_sample_reviews(df: pd.DataFrame, review_col_name: str, number_of_reviews: int = 3):
    """Include reviews and ratings

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    review_col_name: `str`
        The specific_column to get the reviews and ratings of
    
    number_of_reviews: `int`
        Number of samples to include


    Return
    ------
    Nothing; instead, print the reviews with ratings
    """


    columns_to_include = [review_col_name, 'star_rating']

    # Initialize an empty list to store dictionaries
    list_of_dicts = []

    # Iterate over the specified columns and retrieve the first three rows
    for row in df[columns_to_include].head(3).to_dict(orient='records'):
        list_of_dicts.append({'star_rating': row['star_rating'], review_col_name: row[review_col_name]})

    for dictionary in list_of_dicts:
        print(dictionary)


def update_data_type(df: pd.DataFrame, col_name: str):
    """Update the data type of the star ratings

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    col_name: `str`
        Column with rating values

    Return
    ------
    df: `pd.DataFrame`
        An updated DataFrame with the new sentiment appened

    """

    valid_ratings = ['1','2','3','4','5']
    star_rating_series = df[col_name].copy()

    # Convert type to strings
    star_rating_series.astype('str')

    # Check valid list and see which of our stars match
    rows = star_rating_series.index
    is_rating_in_valid_ratings = rows[star_rating_series.isin(valid_ratings)]

    # Convert to list
    is_rating_in_valid_ratings = is_rating_in_valid_ratings.to_list()

    updated_df = df.iloc[is_rating_in_valid_ratings]
    return updated_df

reviews_ratings_df = update_data_type(reviews_ratings_df, 'star_rating')
print("# reviews per rating", reviews_ratings_df['star_rating'].value_counts())
print()


def separate_reviews_by_rating(df: pd.DataFrame, rating_col: str, threshold: int, sentiment_type: str):
    """Categorizes reviews by adding a rating

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    rating_col: `str`
        Column with rating values
    
    threshold: `int`
        Where to split the ratings such that categories can be formed

    sentiment_type: `str`
        One of three types of sentiment: positive, negative, or neural

    Return
    ------
    df: `pd.DataFrame`
        An updated DataFrame with the new sentiment appened
    """


    if sentiment_type == 'positive_sentiment':
        positive_review_threshold = df[rating_col].astype('int32') > threshold
        df = df[positive_review_threshold]
        df[sentiment_type] = 1

    elif sentiment_type == 'negative_sentiment':
        positive_review_threshold = df[rating_col].astype('int32') < threshold
        df = df[positive_review_threshold]
        df[sentiment_type] = 0

    elif sentiment_type == 'neutral_sentiment':
        positive_review_threshold = df[rating_col].astype('int32') == threshold
        df = df[positive_review_threshold]
        df[sentiment_type] = 3
        
    return df

positive_sentiment_df = separate_reviews_by_rating(reviews_ratings_df, 'star_rating', 3, 'positive_sentiment')
print("# positive sentiment: ", len(positive_sentiment_df))
print()
negative_sentiment_df = separate_reviews_by_rating(reviews_ratings_df, 'star_rating', 3, 'negative_sentiment')
print("# negative sentiment: ", len(negative_sentiment_df))
print()
neutral_sentiment_df = separate_reviews_by_rating(reviews_ratings_df, 'star_rating', 3, 'neutral_sentiment')
print("# neutral sentiment: ", len(neutral_sentiment_df))
print()
pos_rand_sampled_df = positive_sentiment_df.sample(100000)
neg_rand_sampled_df = negative_sentiment_df.sample(100000)
reviews_ratings_df = pd.concat([pos_rand_sampled_df, neg_rand_sampled_df])
pos_sentiment = reviews_ratings_df['positive_sentiment'].dropna()
neg_sentiment = reviews_ratings_df['negative_sentiment'].dropna()
reviews_ratings_df['sentiment'] = pd.concat([pos_sentiment, neg_sentiment])
reviews_sentiment_df = reviews_ratings_df.drop(columns=['positive_sentiment', 'negative_sentiment'])
reviews_sentiment_df['review_body'].fillna(' ', inplace=True)

print("Base review body:")
generate_sample_reviews(reviews_sentiment_df, 'review_body', 3)
print()

def convert_reviews_to_lower_case(df: pd.DataFrame, col_name: str):
    """Convert all reviews to lower case

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    col_name: `str`
        Column with reviews

    Return
    ------
    df: `pd.DataFrame`
        An updated DataFrame with the lower cased reviews
    """
    
    lower_case_reviews = []
    updated_df = df.copy()
    text_reviews = df[col_name].values
    
    for text_reviews_idx in range(len(text_reviews)):
        text_review = text_reviews[text_reviews_idx]
        # print(text_reviews_idx, type(text_review), text_review)

        # NOT all reviews are strings, thus all can't be converted to lower cased
        if type(text_review) != str:
            converted_str = str(text_review)
            # update_text_review = converted_str.lower()
            lower_case_reviews.append(text_review)
            # print(text_reviews_idx, update_text_review)
            # print()
        else:
            update_text_review = text_review.lower()
            lower_case_reviews.append(update_text_review)
            # print(text_reviews_idx, update_text_review)
            # print()

    updated_df['lower_cased'] = lower_case_reviews
    return updated_df

reviews_lower_cased = convert_reviews_to_lower_case(reviews_sentiment_df, 'review_body')
print("reviews_lower_cased:")
generate_sample_reviews(reviews_lower_cased, 'lower_cased', 3)
print()

def remove_html_and_urls(df: pd.DataFrame, col_name: str):
    """Remove HTML and URLs from all reviews

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    col_name: `str`
        Column with reviews

    Return
    ------
    df: `pd.DataFrame`
        An updated DataFrame with the html_and_urls removed
    """
    
    # url_pattern = re.compile(r'https?://\S+|www\. \S+')

    cleaned_reviews = []
    updated_df = df.copy()
    text_reviews = df[col_name].values

    for text_reviews_idx in range(len(text_reviews)):
        text_review = text_reviews[text_reviews_idx]

        if isinstance(text_review, str):
            # Check and remove HTML tags
            has_html = bool(re.search('<.*?>', text_review))
            if has_html == True:
                # print("Review", text_reviews_idx, "has HTML -- ", text_review)
                pass

            no_html_review = re.sub('<.*?>', ' ', text_review)
            # print("Review", text_reviews_idx, "without HTML -- ", no_html_review)
        
            # Check and remove URLs
            has_url = bool(re.search(r'http\S+', no_html_review))
            if has_url == True:
                # print("Review", text_reviews_idx, "has URL --", no_html_review)
                pass

            no_html_url_review = re.sub(r'http\S+', '', no_html_review)
            # print("Review", text_reviews_idx, "without HTML, URL -- ", no_html_url_review)
            # print()
            cleaned_reviews.append(no_html_url_review)
        else:
            # print(text_reviews_idx, text_review)
            cleaned_reviews.append(text_review)
            

    updated_df['without_html_urls'] = cleaned_reviews
    return updated_df

no_html_urls_df = remove_html_and_urls(reviews_lower_cased, 'lower_cased')
print("without_html_urls:")
generate_sample_reviews(no_html_urls_df, 'without_html_urls', 3)
print()

store_contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he's": "he is",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "wasn't": "was not",
    "we're": "we are",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    "you're": "you are",
    "you'll": "you will",
    "you'd": "you would",
    "we'll": "we will",
    "we've": "we have",
    "we'd": "we would",
    "I'm": "I am",
    "i've": "I have",
    "I've": "I have",
    "I'd": "I would",
    "it'll": "it will",
    "they'll": "they will",
    "they've": "they have",
    "they'd": "they would",
    "he'll": "he will",
    "he'd": "he would",
    "she'll": "she will",
    "we'd": "we would",
    "we'll": "we will",
    "you've": "you have",
    "you'd": "you would",
    "you'll": "you will",
    "I'll": "I will",
    "I'd": "I would",
    "it's": "it is",
    "it'd": "it would",
    "i'm": "I am",
    "he's": "he is",
    "he'll": "he will",
    "she's": "she is",
    "she'll": "she will",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "that's": "that is",
    "that'll": "that will",
    "that'd": "that would",
    "who's": "who is",
    "who'll": "who will",
    "who'd": "who would",
    "what's": "what is",
    "what'll": "what will",
    "what'd": "what would",
    "when's": "when is",
    "when'll": "when will",
    "when'd": "when would",
    "where's": "where is",
    "where'll": "where will",
    "where'd": "where would",
    "why's": "why is",
    "why'll": "why will",
    "why'd": "why would",
    "how's": "how is",
    "how'll": "how will",
    "how'd": "how would"
}

def locate_and_replace_contractions(review):
    """Find the contractions to replace from a specific review

    Parameters
    ----------
    review: `str`
        A specific review

    Return
    ------
    non_contraction_review: `str`
        The updated specific review with contractions expanded
    
    """
    if isinstance(review, str):
        get_words = review.split()

        store_non_contraction_words = []

        for word in get_words:
            if word in store_contractions:
                non_contraction_form = store_contractions[word]
                # print(word, "-->", non_contraction_form)

                store_non_contraction_words.append(non_contraction_form)

            else:
                # print(word)
                store_non_contraction_words.append(word)

        non_contraction_review = ' '.join(store_non_contraction_words)
        return non_contraction_review
    else:
        return review

def remove_contractions(df:pd.DataFrame, col_name: str):
    """Remove contractions from all reviews

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    col_name: `str`
        Column with reviews

    Return
    ------
    df: `pd.DataFrame`
        An updated DataFrame with the extra spaces removed
    """
    
    without_contractions_reviews = []
    updated_df = df.copy()
    text_reviews = df[col_name].values

    for text_reviews_idx in range(len(text_reviews)):
        text_review = text_reviews[text_reviews_idx]

        # print("Review", text_reviews_idx, "with possible contraction(s) -- ", text_review)

        without_contraction = locate_and_replace_contractions(text_review)

        # print("Review", text_reviews_idx, "without contraction -- ", without_contraction)
        # print()

        without_contractions_reviews.append(without_contraction)

    updated_df['without_contractions'] = without_contractions_reviews
    return updated_df

no_contractions_df = remove_contractions(no_html_urls_df, 'without_html_urls')
print("without_contractions:")
generate_sample_reviews(no_contractions_df, 'without_contractions', 3)
print()

def remove_non_alphabetical_characters(df:pd.DataFrame, col_name: str):
    """Remove Non-alphabetical characters from all reviews

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    col_name: `str`
        Column with reviews

    Return
    ------
    df: `pd.DataFrame`
        An updated DataFrame with the non-alphabetical characters removed
    """

    alphabetical_char_reviews = []
    updated_df = df.copy()
    text_reviews = df[col_name].values
    # print(text_reviews)

    for text_reviews_idx in range(len(text_reviews)):
        text_review = text_reviews[text_reviews_idx]
        
        if isinstance(text_review, str):

            # Check for non-alphabetical characters
            has_non_alphabetical_char = bool(re.search(r'[^a-zA-Z]', text_review))
            if has_non_alphabetical_char == True:
                # print("Review", text_reviews_idx, "has HTML -- ", text_review)
                pass
            
            # Remove non-alphabetical characters
            with_alphabetical_char = re.sub(r'[^a-zA-Z\s]', ' ', text_review)
            # print("Review", text_reviews_idx, "has HTML -- ", with_alphabetical_char)
            alphabetical_char_reviews.append(with_alphabetical_char)
        else:
            alphabetical_char_reviews.append(text_review)

    updated_df['with_alpha_chars_only'] = alphabetical_char_reviews
    return updated_df
only_alpha_chars_df = remove_non_alphabetical_characters(no_contractions_df, 'without_contractions')
print("with_alpha_chars_only:")
generate_sample_reviews(only_alpha_chars_df, 'with_alpha_chars_only', 3)
print()


def remove_extra_spaces(df:pd.DataFrame, col_name: str):
    """Remove extra spaces from all reviews

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    col_name: `str`
        Column with reviews

    Return
    ------
    df: `pd.DataFrame`
        An updated DataFrame with the extra spaces removed
    """
    
    single_spaced_reviews = []
    updated_df = df.copy()
    text_reviews = df[col_name].values
    # print(text_reviews)

    for text_reviews_idx in range(len(text_reviews)):
        text_review = text_reviews[text_reviews_idx]

        if isinstance(text_review, str):
        # Check if there are any extra spaces
            has_extra_space = bool(re.search(r' +', text_review))
            if has_extra_space == True:
                # print("Review", text_reviews_idx, "has extra space -- ", text_review)
                pass
            
            # Remove extra spaces
            single_spaced_review = re.sub(r' +', ' ', text_review)
            # print("Review", text_reviews_idx, "without extra space -- ", single_spaced_review)
            # print()
            
            single_spaced_reviews.append(single_spaced_review)
        else:
            single_spaced_reviews.append(text_review)

    updated_df['without_extra_space'] = single_spaced_reviews
    return updated_df

no_extra_space_df = remove_extra_spaces(only_alpha_chars_df, 'with_alpha_chars_only')
print("without_extra_space:")
generate_sample_reviews(no_extra_space_df, 'without_extra_space', 3)
print()

average_length_after_cleaning = no_extra_space_df['review_body'][no_extra_space_df['review_body'].apply(type) == str].str.len().mean()
print("Average length of the reviews in terms of character length AFTER cleaning", average_length_after_cleaning)
print()

def filter_stop_words(df:pd.DataFrame, col_name: str):
    """Filter stop words out from all reviews

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    col_name: `str`
        Column with reviews

    Return
    ------
    df: `pd.DataFrame`
        An updated DataFrame with the extra spaces removed
    """
    
    without_stop_words_reviews = []
    updated_df = df.copy()
    text_reviews = df[col_name].values

    stop_words = set(stopwords.words("english"))

    for text_reviews_idx in range(len(text_reviews)):
        text_review = text_reviews[text_reviews_idx]

        if isinstance(text_review, str):
            text_review_words = word_tokenize(text_review) 

        

            # print("Before stop word removal", text_reviews_idx, " -- ", text_review)

            filtered_review = []

            for text_review_words_idx in range(len(text_review_words)):
                text_review_word = text_review_words[text_review_words_idx]
                
                # Check if review word is a stop word
                if text_review_word in stop_words:
                    # print("  Stop word -- ", text_review_word)
                    pass
                else:
                    # print(text_review_word, " -- is NOT a stop word in review")
                    filtered_review.append(text_review_word)

            
            filtered_review = " ".join(filtered_review)
            # print("After stop word removal", text_reviews_idx, " -- ", filtered_review)
            # print()
            
            without_stop_words_reviews.append(filtered_review)
        else:
            without_stop_words_reviews.append(text_review)
        

    updated_df['without_stop_words'] = without_stop_words_reviews
    return updated_df

no_stop_words_df = filter_stop_words(no_extra_space_df, 'without_extra_space')
print("without_stop_words:")
generate_sample_reviews(no_stop_words_df, 'without_stop_words', 3)
print()

def lemmentize_review(df:pd.DataFrame, col_name: str):
    """Lemmentize all reviews

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    col_name: `str`
        Column with reviews

    Return
    ------
    df: `pd.DataFrame`
        An updated DataFrame with the extra spaces removed
    """
    
    lemmed_reviews = []
    updated_df = df.copy()
    text_reviews = df[col_name].values

    lem = WordNetLemmatizer()

    for text_reviews_idx in range(len(text_reviews)):
        text_review = text_reviews[text_reviews_idx]   
        if isinstance(text_review, str):     
            words_in_review = word_tokenize(text_review) 

            # print("Before lem update", text_reviews_idx, " -- ", text_review)
            # print("Lemmed words", words_in_review)
            

            lemmed_sentence = []

            # Split review into words
            for lemmed_words_idx in range(len(words_in_review)):
                word = words_in_review[lemmed_words_idx]
                
                apply_lemmatization = lem.lemmatize(word)
                # print(apply_lemmatization)
                
                lemmed_sentence.append(apply_lemmatization)
                filtered_review = " ".join(lemmed_sentence)
        
            # print("After lem update -- ", filtered_review)
            # print()

            lemmed_reviews.append(filtered_review)
        else:
            lemmed_reviews.append(text_review)

    updated_df['lemmed_reviews'] = lemmed_reviews
    return updated_df

lemmed_df = lemmentize_review(no_stop_words_df, 'without_stop_words')
print("without_stop_words:")
generate_sample_reviews(lemmed_df, 'lemmed_reviews', 3)
print()

def tf_idf_feature_extraction(df: pd.DataFrame, col_name: str):
    """Extract the TF-IDF features from the reviews.

    Parameters
    ----------
    df: `pd.DataFrame`
        The data
    
    col_name: `str`
        Column with reviews

    Return
    ------
    tf_idf_features:
        A matrix containing the TF-IDF features extracted
        
    """
    
    vectorizer = TfidfVectorizer()
    tf_idf_features = vectorizer.fit_transform(df[col_name])

    return tf_idf_features

tf_idf_features = tf_idf_feature_extraction(lemmed_df, 'lemmed_reviews')
sentiments = lemmed_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(tf_idf_features, sentiments, test_size=0.2, random_state=42)
print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print()

def eval_accuracy(y_true, y_prediction):
    return sklearn.metrics.accuracy_score(y_true, y_prediction)

def eval_precision(y_true, y_prediction):
    return sklearn.metrics.precision_score(y_true, y_prediction)

def eval_recall(y_true, y_prediction):
    return sklearn.metrics.recall_score(y_true, y_prediction)

def eval_f1_score(y_true, y_prediction):
    return sklearn.metrics.f1_score(y_true, y_prediction)


def train_eval_metric(y_train_true, y_train_predictions):
    accuracy = eval_accuracy(y_train_true, y_train_predictions)
    precision = eval_precision(y_train_true, y_train_predictions)
    recall = eval_recall(y_train_true, y_train_predictions)
    f1 = eval_f1_score(y_train_true, y_train_predictions)

    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    return metrics_dict

def test_eval_metric(y_test_true, y_test_predictions):
    accuracy = eval_accuracy(y_test_true, y_test_predictions)
    precision = eval_precision(y_test_true, y_test_predictions)
    recall = eval_recall(y_test_true, y_test_predictions)
    f1 = eval_f1_score(y_test_true, y_test_predictions)

    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    return metrics_dict

def perceptron_model(X_train, X_test, y_train, y_test): 

    technique = Perceptron(tol=1e-3, random_state=0)
    technique.fit(X_train, y_train)
    y_train_predictions = technique.predict(X_train)
    y_test_predictions = technique.predict(X_test)


    train_metrics = train_eval_metric(y_train, y_train_predictions)
    test_metrics = test_eval_metric(y_test, y_test_predictions)

    return train_metrics, test_metrics

perceptron_train_metrics, perceptron_test_metrics = perceptron_model(X_train, X_test, y_train, y_test)
print("Perceptron:", "Train", perceptron_train_metrics, "Test", perceptron_test_metrics)
print()

def svm_model(X_train, X_test, y_train, y_test): 

    technique = LinearSVC(tol=1e-3, random_state=0)
    technique.fit(X_train, y_train)
    y_train_predictions = technique.predict(X_train)
    y_test_predictions = technique.predict(X_test)


    train_metrics = train_eval_metric(y_train, y_train_predictions)
    test_metrics = test_eval_metric(y_test, y_test_predictions)

    return train_metrics, test_metrics
svm_train_metrics, svm_test_metrics = svm_model(X_train, X_test, y_train, y_test)
print("SVM:", "Train", svm_train_metrics, "Test", svm_test_metrics)
print()

def logistic_regression_model(X_train, X_test, y_train, y_test): 

    technique = LogisticRegression(random_state=0)
    technique.fit(X_train, y_train)
    y_train_predictions = technique.predict(X_train)
    y_test_predictions = technique.predict(X_test)


    train_metrics = train_eval_metric(y_train, y_train_predictions)
    test_metrics = test_eval_metric(y_test, y_test_predictions)

    return train_metrics, test_metrics

logistic_regression_train_metrics, logistic_regression_test_metrics = logistic_regression_model(X_train, X_test, y_train, y_test)
print("Logistic Regression:", "Train", logistic_regression_train_metrics, "Test", logistic_regression_test_metrics)

def naive_bayes_model(X_train, X_test, y_train, y_test): 

    technique = MultinomialNB()
    technique.fit(X_train.toarray(), y_train)
    y_train_predictions = technique.predict(X_train)
    y_test_predictions = technique.predict(X_test)

    train_metrics = train_eval_metric(y_train, y_train_predictions)
    test_metrics = test_eval_metric(y_test, y_test_predictions)

    return train_metrics, test_metrics

naive_bayes_train_metrics, naive_bayes_test_metrics = naive_bayes_model(X_train, X_test, y_train, y_test)
print("Naive Bayes", "Train", naive_bayes_train_metrics, "Test", naive_bayes_test_metrics)
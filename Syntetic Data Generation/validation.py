# pip install nltk 
# pip install textblob 
# pip install flair 
# pip install transformers

import json

import nltk
import pandas as pd
from flair.data import Sentence
from flair.models import TextClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline

# Download the vader_lexicon data
nltk.download("vader_lexicon")
# Create a sentiment analyzer using NTLK's Vader
analyzer = SentimentIntensityAnalyzer()
# Load twitter-roberta-base-sentiment-latest model for sentiment analysis
sentiment_classifier = pipeline("sentiment-analysis",
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest")
# Load a pre-trained sentiment analysis model from Flair
classifier = TextClassifier.load('en-sentiment')


def get_reviews():
    with open('generated_text_positive.json') as f:
        return json.load(f)


def classify_sentiment(polarity, polarity_threshold=0.05):
    if polarity >= polarity_threshold:
        return "Positive"
    elif polarity <= -polarity_threshold:
        return "Negative"
    else:
        return "Neutral"


def classify_sentiment_ntlk(review: str):
    """
    Classify sentiment using NTLK's Vader
    """
    scores = analyzer.polarity_scores(review)
    return classify_sentiment(scores['compound'])


def classify_sentiment_textblob(review: str):
    """
    Classify sentiment using TextBlob
    """
    blob = TextBlob(review)
    return classify_sentiment(blob.sentiment.polarity)


def classify_sentiment_flair(review):
    """
    Classify sentiment using Flair
    """
    sentence = Sentence(review)
    classifier.predict(sentence)
    return sentence.labels[0].value


def classify_sentiment_twitter_roberta_base(review):
    """
    Classify sentiment using Twitter-roBERTa-base
    """
    sentiment = sentiment_classifier(review)
    return sentiment[0]['label']


def get_sentiments(reviews):
    results = []
    for review in reviews:
        result = {
            "Review": review,
            "NTLK": classify_sentiment_ntlk(review),
            "TextBlob": classify_sentiment_textblob(review),
            "Flair": classify_sentiment_flair(review),
            "Twitter_roBERTa": classify_sentiment_twitter_roberta_base(review),
        }
        results.append(result)

    # Return the results as a pandas DataFrame
    return pd.DataFrame(results)


def main():
    reviews = get_reviews()
    results = get_sentiments(reviews)
    print("Sentiment results: ", results)


if __name__ == '__main__':
    main()

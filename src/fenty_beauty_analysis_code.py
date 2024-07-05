import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ntscraper import Nitter

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize Nitter
scraper = Nitter(log_level=1, skip_instance_check=False)

# Define a function to scrape tweets and return a DataFrame
def get_tweets(query, mode, size):
    tweets = scraper.get_tweets(query, mode=mode, number=size)
    final_tweets = []
  
    for tweet in tweets['tweets']:
        data = [tweet['link'], tweet['text'], tweet['date'], tweet['stats']['likes'], tweet['stats']['comments']]
        final_tweets.append(data)
  
    data = pd.DataFrame(final_tweets, columns=['Link', 'Text', 'Date', 'Likes', 'Comments'])
    return data

# Collect tweets
query = "fenty beauty"
max_tweets = 50  # Adjust based on your requirements
tweets_df = get_tweets(query, mode="term", size=max_tweets)

# Check if tweets_df is empty
if tweets_df.empty:
    print("No tweets were scraped.")
else:
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Apply Sentiment Analysis
    tweets_df['Sentiment'] = tweets_df['Text'].apply(lambda text: sid.polarity_scores(text)['compound'])

    # Classify Sentiments
    def classify_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    tweets_df['Sentiment_Class'] = tweets_df['Sentiment'].apply(classify_sentiment)

    # Plot Sentiment Distribution
    sentiment_counts = tweets_df['Sentiment_Class'].value_counts()

    # Check if sentiment_counts is empty
    if sentiment_counts.empty:
        print("No sentiments were classified.")
    else:
        sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
        plt.title('Sentiment Analysis of Fenty Beauty Tweets')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.show()

import pandas as pd
import textblob
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from the CSV file.
    """

    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Dropping unnecessary columns and renaming the columns.
    """
    del df['ProfileName'] # ProfileName is not useful for analysis and it contains a lot of null values
    del df['UserId'] # UserId is not useful for analysis
    del df['Summary'] # Summary provides a shortened version of the review, meaning keywords for the seniment analysis are not present,
    # it also contains a lot of null values
    del df['Time'] # Time is not useful for analysis

    # Renaming columns for better readability
    df.rename(columns={'Id':'review_id', 'ProductId':'product_id', 'HelpfulnessNumerator':'helpfulness_numerator', 'Score':'product_rating', 'Text':'review_text'}, inplace=True)

    return df

def textblob_scoring(df_sample):
    """
    Perform sentiment analysis using TextBlob.
    """

    scores = []

    for row in df_sample.review_text:

        total_score = 0
        count = 0
        #blob = textblob.TextBlob(row['review_text'])
        blob = textblob.TextBlob(row)

        for sentence in blob.sentences:
            total_score += sentence.sentiment.polarity
            count += 1

        scores.append(total_score/count)

    df_sample['textblob_score'] = scores
    return df_sample

def sentiment_classification(df_sample):
    """
    Classify the sentiment based on the TextBlob score.
    """
    df_sample['sentiment'] = df_sample['textblob_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    return df_sample

def collocation_extraction_pmi(df_sample):
    """
    Extract collocations from the review text.
    """
    # Tokenize the review text
    tokens = nltk.word_tokenize(' '.join(df_sample['review_text']))
    # Extract collocations using NLTK's BigramCollocationFinder
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(3)  # Filter out bigrams that occur less than 3 times
    collocations = finder.nbest(bigram_measures.pmi, 10)  # Get top 10 collocations based on Pointwise Mutual Information (PMI)
    return collocations

def main():
    """
    Main function to execute the script.
    """
    df = load_data("Reviews.csv")
    df = preprocess_data(df)
    print('Processing complete.')
    print(df.head())

    # Sample 10,000 rows for analysis
    df_sample = df.sample(10000, random_state=42)
    # Storing the sampled data to a CSV file
    df_sample.to_csv("sampled_reviews.csv", index=False)

    textblob_scoring(df_sample)
    df_sample = sentiment_classification(df_sample)
    print(df_sample)

    collocation_extraction_pmi(df_sample)


main()
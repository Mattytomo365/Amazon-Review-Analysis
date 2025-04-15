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
    del df['ProfileName'] # ProfileName is not useful for analysis
    del df['UserId'] # UserId is not useful for analysis
    del df['Summary'] # Summary provides a shortened version of the review, meaning keywords for the seniment analysis are not present
    del df['Time'] # Time is not useful for analysis

    # Renaming columns for better readability
    df.rename(columns={'Id':'review_id', 'ProductId':'product_id', 'HelpfulnessNumerator':'helpfulness_numerator', 'Score':'product_rating', 'Text':'review_text'}, inplace=True)

    return df

def textblob_scoring(df):
    """
    Perform sentiment analysis using TextBlob.
    """

    scores = []

    for row in df.iterrows():

        total_score = 0
        count = 0
        blob = textblob.TextBlob(row['review_text'])

        for sentence in blob.sentences:
            total_score += sentence.sentiment.polarity
            count += 1

        scores.append(total_score/count)

    df['textblob_score'] = scores
    return df

def main():
    """
    Main function to execute the script.
    """
    df = load_data("Reviews.csv")
    df = preprocess_data(df)
    textblob_scoring(df)
    print(df)

main()
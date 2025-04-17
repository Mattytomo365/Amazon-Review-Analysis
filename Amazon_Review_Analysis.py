import pandas as pd
import textblob
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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

def collocation_extraction_co_occurrence(df_sample, sentiment, sentiment_filtered=True, pos_filtered=True):
    """
    Extract collocations and co-occurrences from the reviews.
    """
    nltk.download('punkt')
    nltk.download('stopwords')

    if sentiment_filtered:
        # Filtering reviews based on sentiment classification
        reviews = df_sample[df_sample['sentiment'] == sentiment]['review_text']

        bigram = Counter()
        unigram = Counter()

        if pos_filtered:
            # Filter tokens based on POS tagging
            tokens = tokens.apply(lambda x: [word for word, pos in nltk.pos_tag(x) if pos.startswith('NN') or pos.startswith('JJ')])

            # Create a list of all tokens
            all_tokens = [token for sublist in tokens for token in sublist]
            for i in range(len(all_tokens)):
                bigram[(all_tokens[i], all_tokens[i + 1])] += 1
                unigram[(all_tokens[i])] += 1

        else:
            # Tokenize the reviews
            tokens = reviews.apply(lambda x: nltk.word_tokenize(x.lower()))
            tokens = reviews.apply(lambda x: [word for word in x if word.isalpha()])

            all_tokens = [token for sublist in tokens for token in sublist]
            for i in range(len(all_tokens)):
                bigram[(all_tokens[i], all_tokens[i + 1])] += 1
                unigram[(all_tokens[i])] += 1


    else:
        reviews = df_sample['review_text']
        # Tokenize the reviews
        tokens = reviews.apply(lambda x: nltk.word_tokenize(x.lower()))
        tokens = reviews.apply(lambda x: [word for word in x if word.isalpha()])

        # Create a list of all tokens
        all_tokens = [token for sublist in tokens for token in sublist]

        for i in range(len(all_tokens)):
            bigram[(all_tokens[i], all_tokens[i + 1])] += 1
            unigram[(all_tokens[i])] += 1

        # Create a frequency distribution of the tokens
        freq_dist = nltk.FreqDist(all_tokens)
        plt.figure(figsize=(12, 6))
        freq_dist.plot(30, cumulative=False)
        plt.show()

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

    collocation_extraction_co_occurrence(df_sample) # take this out before merging


main()
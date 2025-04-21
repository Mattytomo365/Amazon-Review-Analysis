import pandas as pd
import textblob
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
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

def collocation_extraction_pmi(df_sample, sentiment, pos_filtered = False):
    """
    Extract collocations from the review text.
    """
    if sentiment == ' ':
        sentiment_filtered = False
    else:
        sentiment_filtered = True

    if sentiment_filtered:
        # Filter the DataFrame based on sentiment
        reviews = df_sample[df_sample['sentiment'] == sentiment]['review_text']

        # Tokenize the review text
        tokens = reviews.apply(lambda x: [word for word in nltk.word_tokenize(x.lower()) 
        if word.isalpha() and word not in string.punctuation and word not in stopwords.words('english')
        ])

        if pos_filtered:
            # Filter tokens based on POS tags
            tokens = tokens.apply(lambda x: [word for word, pos in nltk.pos_tag(x) if pos.startswith('NN') or pos.startswith('JJ')])
            # Extract collocations using NLTK's BigramCollocationFinder
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
            finder.apply_freq_filter(3)  # Filter out bigrams that occur less than 3 times
            collocations = finder.nbest(bigram_measures.pmi, 10)  # Get top 10 collocations based on Pointwise Mutual Information (PMI)
            print(f'Top 10 collocations (pointwise mutual information)({sentiment} filtered)(pos tag filtered):')

        else:
            # Extract collocations using NLTK's BigramCollocationFinder
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
            finder.apply_freq_filter(3)
            collocations = finder.nbest(bigram_measures.pmi, 10)
            print(f'Top 10 collocations (pointwise mutual information)({sentiment} filtered)(pos tag unfiltered):')

    else:
        # Establishing text to tokenise
        reviews = df_sample['review_text']

        tokens = reviews.apply(lambda x: [word for word in nltk.word_tokenize(x.lower()) 
        if word.isalpha() and word not in string.punctuation and word not in stopwords.words('english')
        ])

        if pos_filtered:
            tokens = tokens.apply(lambda x: [word for word, pos in nltk.pos_tag(x) if pos.startswith('NN') or pos.startswith('JJ')])
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
            finder.apply_freq_filter(3)
            collocations = finder.nbest(bigram_measures.pmi, 10)
            print('Top 10 collocations (pointwise mutual information)(sentiment unfiltered)(pos tag filtered):')
        else:
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
            finder.apply_freq_filter(3)
            collocations = finder.nbest(bigram_measures.pmi, 10)
            print('Top 10 collocations (pointwise mutual information)(sentiment unfiltered)(pos tag filtered):')
    
    collocation_data = []

    for bigram in collocations:
        for (word1, word2) in bigram:
            collocation_data.append((bigram, word1, word2))
            

    collocations_df = pd.DataFrame(collocation_data, columns=['Collocation', 'Word1', 'Word2'])

    print(collocations_df)

def main():
    """
    Main function to execute the script.
    """
    df = load_data("Reviews.csv")
    df = preprocess_data(df)
    print('Processing complete.')
    print(df.head())

    # Sample 10,000 rows for analysis
    df_sample = df.sample(1000, random_state=42)
    # Storing the sampled data to a CSV file
    df_sample.to_csv("sampled_reviews.csv", index=False)

    textblob_scoring(df_sample)
    df_sample = sentiment_classification(df_sample)
    print(df_sample)

    collocation_extraction_pmi(df_sample, 'positive') # take this out before merging



main()
import pandas as pd
import textblob
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
from nltk.corpus import stopwords
import textwrap

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


def collocation_extraction_co_occurrence(df_sample, sentiment, pos_filtered=True):
    """
    Extract collocations and co-occurrences from the reviews.
    """
    if sentiment == ' ':
        sentiment_filtered = False
    else:
        sentiment_filtered = True

    if sentiment_filtered:
        # Filtering reviews based on sentiment classification
        reviews = df_sample[df_sample['sentiment'] == sentiment]['review_text']

        bigram = Counter()
        unigram = Counter()

        tokens = reviews.apply(lambda x: [word for word in nltk.word_tokenize(x.lower()) 
                if word.isalpha() and word not in string.punctuation and word not in stopwords.words('english')
        ])


        if pos_filtered:
            # Filter tokens based on POS tagging
            tokens = tokens.apply(lambda x: [word for word, pos in nltk.pos_tag(x) if pos.startswith('NN') or pos.startswith('JJ')])

            # Create a list of all tokens
            all_tokens = [token for sublist in tokens for token in sublist]
            for i in range(len(all_tokens) - 1):
                bigram[(all_tokens[i], all_tokens[i + 1])] += 1
                unigram[(all_tokens[i])] += 1

        else:
            # Tokenize the reviews
            tokens = tokens.apply(lambda x: [word for word in x if word.isalpha()])

            all_tokens = [token for sublist in tokens for token in sublist]
            for i in range(len(all_tokens) - 1):
                bigram[(all_tokens[i], all_tokens[i + 1])] += 1
                unigram[(all_tokens[i])] += 1

    else:
        # Establishing text to tokenise
        reviews = df_sample['review_text']

        bigram = Counter()
        unigram = Counter()

        tokens = reviews.apply(lambda x: [word for word in nltk.word_tokenize(x.lower()) 
                if word.isalpha() and word not in string.punctuation and word not in stopwords.words('english')
        ])

        if pos_filtered:
            # Filter tokens based on POS tagging
            tokens = tokens.apply(lambda x: [word for word, pos in nltk.pos_tag(x) if pos.startswith('NN') or pos.startswith('JJ')])

            # Create a list of all tokens
            all_tokens = [token for sublist in tokens for token in sublist]
            for i in range(len(all_tokens) - 1):
                bigram[(all_tokens[i], all_tokens[i + 1])] += 1
                unigram[(all_tokens[i])] += 1
        
        else:  
            reviews = df_sample['review_text']

            # Create a list of all tokens
            all_tokens = [token for sublist in tokens for token in sublist]

            for i in range(len(all_tokens) - 1):
                bigram[(all_tokens[i], all_tokens[i + 1])] += 1
                unigram[(all_tokens[i])] += 1

    collocation_data = []

    for (word1, word2), bigram_count in bigram.items():
        unigram_count_word1 = unigram[word1]
        unigram_count_word2 = unigram[word2]
        collocation_data.append((word1, word2, bigram_count, unigram_count_word1, unigram_count_word2))

    collocations_df = pd.DataFrame(collocation_data, columns=['Word1', 'Word2', 'Co-occurrence', 'Word1_Count', 'Word2_Count'])
    collocations_df = collocations_df.sort_values(by='Co-occurrence', ascending=False).head(10)

    if pos_filtered:
        co_occurrence_table(collocations_df, sentiment)
    else:
        co_occurrence_table(collocations_df, sentiment, False)


def co_occurrence_table(df_sample, sentiment, pos_filtered=True):
    """
    Displaying collocations in a table format (Co-Occurrence Approach).
    """
    if sentiment == ' ':
        sentiment_filtered = False
    else:
        sentiment_filtered = True

    if sentiment_filtered:
        if pos_filtered:
            title = f"Top 10 Collocations in {sentiment} Reviews (Co-Occurrence Approach)(POS Tag Filtered)"
        else:
            title = f"Top 10 Collocations in {sentiment} Reviews (Co-Occurrence Approach)(POS Tag Unfiltered)"
    else:
        title = "Top 10 Collocations in All Reviews (Co-Occurrence Approach)"

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    table = ax.table(cellText=df_sample.values, colLabels=['First Word', 'Second Word', 'Co-Occurrence', 'Word 1 Freq', 'Word 2 Freq'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    plt.subplots_adjust(top=0.95)
    plt.show()

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

        # Flatten the list of lists into a single list
        flat_tokens = [word for review in tokens for word in review]

        if pos_filtered:
            # Filter tokens based on POS tags
            tokens = tokens.apply(lambda x: [word for word, pos in nltk.pos_tag(x) if pos.startswith('NN') or pos.startswith('JJ')])
            # Assuming tokens is a list of lists, e.g. from df['tokens']
            flat_tokens = [word for review in tokens for word in review]
            # Extract collocations using NLTK's BigramCollocationFinder
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(flat_tokens)
            finder.apply_freq_filter(3)  # Filter out bigrams that occur less than 3 times
            collocations = finder.nbest(bigram_measures.pmi, 10)  # Get top 10 collocations based on Pointwise Mutual Information (PMI)

        else:
            # Extract collocations using NLTK's BigramCollocationFinder
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(flat_tokens)
            finder.apply_freq_filter(3)
            collocations = finder.nbest(bigram_measures.pmi, 10)

    else:
        # Establishing text to tokenise
        reviews = df_sample['review_text']

        tokens = reviews.apply(lambda x: [word for word in nltk.word_tokenize(x.lower()) 
        if word.isalpha() and word not in string.punctuation and word not in stopwords.words('english')
        ])

        if pos_filtered:
            tokens = tokens.apply(lambda x: [word for word, pos in nltk.pos_tag(x) if pos.startswith('NN') or pos.startswith('JJ')])
            flat_tokens = [word for review in tokens for word in review]
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(flat_tokens)
            finder.apply_freq_filter(3)
            collocations = finder.nbest(bigram_measures.pmi, 10)

        else:
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(flat_tokens)
            finder.apply_freq_filter(3)
            collocations = finder.nbest(bigram_measures.pmi, 10)
    
    collocation_data = []


    for (word1, word2) in collocations:
        bigram = word1 + ' ' + word2
        collocation_data.append((bigram, word1, word2))
            
    collocations_df = pd.DataFrame(collocation_data, columns=['Collocation', 'Word1', 'Word2'])

    if pos_filtered:
        pmi_table(collocations_df, sentiment)
    else:
        pmi_table(collocations_df, sentiment, False)


def pmi_table(df_sample, sentiment, pos_filtered=True):
    """
    Displaying collocations in a table format (PMI Approach).
    """
    if sentiment == ' ':
        sentiment_filtered = False
    else:
        sentiment_filtered = True

    if sentiment_filtered:
        if pos_filtered:
            title = f"Top 10 Collocations in {sentiment} Reviews (PMI Approach)(POS Tag Filtered)"
        else:
            title = f"Top 10 Collocations in {sentiment} Reviews (PMI Approach)(POS Tag Unfiltered)"
    else:
        title = "Top 10 Collocations in All Reviews (PMI Approach)"

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    table = ax.table(cellText=df_sample.values, colLabels=['Collocation', 'Word 1', 'Word 2'], loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    plt.subplots_adjust(top=0.95)
    plt.show()


def sentiment_totals(df_sample):
    """
    Calculate the total number of reviews for each sentiment category.
    """
    totals = []
    classifications = ['positive', 'negative', 'neutral']

    for classification in classifications:
        total = df_sample[df_sample['sentiment'] == classification].shape[0]
        totals.append(total)

    print('Total number of reviews for each sentiment category:')
    
    df_sentiment_totals = pd.DataFrame({'Classification' : classifications, 'Total Reviews': totals})
    print(df_sentiment_totals)

    # Plotting the sentiment totals
    plt.figure(figsize=(10, 6))
    plt.pie(df_sentiment_totals['Total Reviews'], labels=df_sentiment_totals['Classification'], autopct='%1.1f%%', startangle=140)
    plt.title('Review Classification Totals')
    plt.legend(title='Classifications')
    plt.show()


def sentiment_distribution(df_sample):
    """
    Plot the distribution of sentiment scores.
    """
    # Boxplots for sentiment score distributions across classifications
    sns.boxplot(x='sentiment', y='textblob_score', data=df_sample)
    plt.title('Distribution of Sentiment Scores (Classification Specific)')
    plt.xlabel('Classification')
    plt.ylabel('Sentiment Score')
    plt.show()

    # Histogram for overall sentiment score distribution
    plt.hist(df_sample['textblob_score'], bins=30, alpha=0.7)
    plt.title('Distribution of Sentiment Scores (Overall)')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()


def top_reviews(df_sample):
    """
    Display the top reviews for each sentiment category.
    """
    positives = df_sample[df_sample['sentiment'] == 'positive']
    negatives = df_sample[df_sample['sentiment'] == 'negative']

    top_positive_reviews = positives.nlargest(5, 'textblob_score')[['review_text', 'textblob_score']]
    top_negative_reviews = negatives.nsmallest(5, 'textblob_score')[['review_text', 'textblob_score']]

    # Internal function used to neatly wrap long review text for display in tables
    def wrap_text(text, width=80):
        return '\n'.join(textwrap.wrap(text, width))
    
    # Apply text wrapping to review content
    top_positive_reviews['review_text'] = top_positive_reviews['review_text'].apply(lambda x: wrap_text(str(x)))
    top_negative_reviews['review_text'] = top_negative_reviews['review_text'].apply(lambda x: wrap_text(str(x)))

    fig, ax = plt.subplots(figsize = (8, 5))
    ax.axis('off')
    ax.set_title('Top 5 Most Positive Reviews', fontsize=14, fontweight="bold")
    table = ax.table(cellText=top_positive_reviews.values, colLabels=['Review Content', 'Sentiment Score'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 3)  # Scale table to fit better
    
    plt.show()
    
    fig, ax = plt.subplots(figsize = (9, 5))
    ax.axis('off')
    ax.set_title('Top 5 Most Negative Reviews', fontsize=14, fontweight="bold")
    table = ax.table(cellText=top_negative_reviews.values, colLabels=['Review Content', 'Sentiment Score'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    table.scale(1, 3)  # Scale table to fit better
    
    plt.show()


def main():
    """
    Main function to execute the script.
    """
    pd.set_option('display.max_columns', None)
    df = load_data("Reviews.csv")
    df = preprocess_data(df)
    print('Processing complete.')
    print(df.head())

    df_sample = df.sample(100, random_state=42) # Change the sample size as needed, the smaller the smaple the faster the code will run.

    # Storing the sampled data to a CSV file
    df_sample.to_csv("sampled_reviews.csv", index=False)

    textblob_scoring(df_sample)
    df_sample = sentiment_classification(df_sample)
    print('Sentiment classification complete.')

    '''
    - If you want to filter the collocations based on sentiment, set sentiment to 'positive', 'negative', or 'neutral'.
    - If you don't want to filter the collocations based on sentiment, set sentiment to ' '.
    - If you want to filter the collocations based on POS tags, set pos_filtered to True.
    - If you don't want to filter the collocations based on POS tags, set pos_filtered to False.
    '''
    collocation_extraction_co_occurrence(df_sample, 'positive', pos_filtered=True) # Co-occurrence extraction approach
    collocation_extraction_pmi(df_sample, 'positive', pos_filtered=True) # Pointwise Mutual Information extraction approach

    sentiment_totals(df_sample)
    sentiment_distribution(df_sample)
    top_reviews(df_sample)

main()
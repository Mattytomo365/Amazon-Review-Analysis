# Amazon Review Analysis

---

## Overview

This project involves analyzing a dataset of Amazon product reviews. The analysis includes sentiment analysis, collocation extraction, and other techniques to derive insights from the data.

Due to the size of the dataset, it is not included in this repository, however, you can download it from **Kaggle** [here](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)

You can track the project's progress [here](https://www.notion.so/1d518110f1f280a2b2c5c5c689bddc9f?v=1d518110f1f2805fa1d8000c4f339738&pvs=4)

---

## Features
- **Preprocessing**: Cleaning and preparing the dataset for analysis.
- **Sentiment Analysis**: Classifying reviews as positive, negative, or neutral.
- **Collocation Extraction**: Identifying frequently co-occurring words using PMI and co-occurrence methods.
- **Visualizations**: Generating insightful visualizations to represent the analysis results.

---

## Project Structure

### Key Files
- `Amazon_Review_Analysis.py`: The main script for performing analysis on the dataset.
- `Reviews.csv`: The original dataset containing Amazon product reviews.
- `sampled_reviews.csv`: A sampled subset of the dataset for testing and development purposes.
- `README.md`: Documentation for the project.

### Branches Overview
- `preprocessing` (merged): Reformatting and cleaning the dataset in preparation for analysis.
- `sentiment-analysis` (merged): Implementing TextBlob scoring and sentiment labelling.
- `performance-optimisation` (merged): Introducing dataset sampling to improve compile time.
- `collocation-extraction-pmi` (merged): Extracting collocations using the Pointwise Mutual Information (PMI) approach.
- `collocation-extraction-co-occurrence` (merged): Extracting collocations using the Co-Occurrence approach.
- `sentiment-visualisations` (merged): Adding visualisations to display sentiment patters.
- `visualisation-improvements` (merged): Improving the visualisation of collocation extraction results.

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Required Python libraries:
    - `pandas`
    - `numpy`
    - `matplotlib`
    - `seaborn`
    - `nltk`
    - `textblob`

### Installation

1. Clone the repository:

    ```
    git clone https://github.com/your-username/Amazon-Review-Analysis.git
    ```

2. Navigate to the project directory:

    ```
    cd Amazon_Review_Analysis
    ```

3. Download necessary NLTK resources:

    ```
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

    ```

### Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) and place it in the project directory.

2. Rename the CSV file to `Reviews.csv`

3. Run the program

4. Customise the collocation extraction filtering options to your needs (refer to inline comments).

---

## Contributing
**Contributions are welcome!**\
Please fork the repository and submit a pull request with your changes.

---

## Contact
For any questions or feedback, feel free to reach out:
- **Email**: matty.tom@icloud.com
- **GitHub**: [Mattytomo365](https://github.com/Mattytomo365)
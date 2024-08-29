# Sentiment Analysis

This is a simple sentiment analysis project that preprocesses and analyzes product reviews from an Amazon dataset.

## Libraries

- pandas - for data manipulation
- textblob - for simple sentiment analysis
- nltk - for text preprocessing (tokenization, stop words, lemmatization)

## Dataset

- `product_reviews_data.csv` - taken from Kaggle [Amazon Product Reviews](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset) dataset.


## Setup

```bash
python3 -m venv env

# mac os
source ./env/bin/activate
# windows (cmd)
.\env\Scripts\activate.bat
# windows (powershell)
.\env\Scripts\Activate.ps1

pip install -r requirements.txt

# you may need to download nltk data, which requires certificates
# mac os (replace 3.x with your python version)
"/Applications/Python 3.x/Install Certificates.command"
# windows
pip install --upgrade certifi
```

## Running the application

```bash
python main.py
```

## Running tests

this command should run all the tests in the `tests` directory.

```bash
pytest
```

## Development Notes

### Approach

My general idea of sentiment analysis is the following:
- preprocess the text
    - tokenization (like a simple version of creating embeddings)
    - remove unnecessary words (stop words)
    - "normalizing" words (lemmatization)
- perform sentiment analysis
    - use `TextBlob` to get sentiment polarity
    - (future) try more advanced sentiment analysis models
- generate outputs
    - visualizations
    - csv output

The bulk of the sentiment analysis logic was done just over an hour. 
I spent another hour adding tests and doing manual checking.
Additionally, around 30 minutes was spent refactoring and modularizing code.

### Testing

I did a reasonable amount of manual checking of the output. 

"Neutral" seems to be inconsistent, so depending on the dataset I had to adjust the values to match what I was seeing. A future plan would be to come up with a better way to dynamically adjust the values based on the data being fed

I really like unit & integration tests, especially working with `pytest`; I set up some accuracy tests for sentiment analysis and feature extraction against pre-determined sample data.

The unit & integration tests helped me catch some mistakes and ensure that I was correctly testing the code. That led me back to manual checking, so the back and forth helps a lot in overall debugging and improving the code.
# NLP Assignments: Spam Detection, Sentiment Analysis, and Topic Modeling

## Overview
This repository contains several NLP projects and experiments, including:
- Spam classification using large language models (LLMs)
- Sentiment analysis on tweets
- Topic modeling on the 20 Newsgroups dataset
- A Streamlit chatbot interface for local LLM inference

## Contents

### Notebooks & Scripts
- **LLM_Usage_part_1.ipynb**: Classifies emails as spam or ham using a Hugging Face LLM (zero-shot classification) on the `spam.csv` dataset. Evaluates accuracy, F1, recall, and confusion matrix.
- **LLM_Usage_part_2.py**: Streamlit app for a local chatbot using the Flan-T5-small model. Allows interactive Q&A with an LLM running locally.
- **Sentiment_analysis_part_1.ipynb**: Performs sentiment analysis on tweets using the VADER sentiment analyzer. Includes text cleaning and labeling.
- **Sentiment_analysis_part_2.ipynb**: Performs sentiment analysis on tweets using a Hugging Face transformer pipeline. Includes text cleaning and batch inference.
- **Topic_modeling.ipynb**: Topic modeling on the newsgroups dataset using LDA and NMF. Visualizes topics with bar charts and word clouds.

### Datasets
- **spam.csv**: SMS messages labeled as `spam` or `ham` (not spam). Used for spam classification.
- **tweets-data.csv**: Tweets with metadata (date, likes, hashtags, etc.). Used for sentiment analysis.
- **newsgroups**: Binary (pickled) file containing the 20 Newsgroups dataset or similar. Used for topic modeling.

## Setup & Requirements
Install the required Python packages:
```bash
pip install transformers pandas tqdm nltk vaderSentiment scikit-learn matplotlib wordcloud streamlit torch
```

For notebooks using NLTK, you may need to download resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage
- Open the Jupyter notebooks (`.ipynb`) for step-by-step code and explanations.
- Run `LLM_Usage_part_2.py` with Streamlit:
  ```bash
  streamlit run LLM_Usage_part_2.py
  ```
- Place the datasets (`spam.csv`, `tweets-data.csv`, `newsgroups`) in the project root.

## Notes
- The `newsgroups` file is binary and should be loaded with Python's `pickle` module.
- Some scripts may require GPU or MPS support for faster inference.

---
Feel free to adapt or extend these notebooks for your own NLP experiments! 

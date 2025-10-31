# News Category Classification
A complete end-to-end notebook for classifying news articles into four categories (World, Sports, Business, Sci/Tech) using the AG News dataset. The project demonstrates data loading, preprocessing (cleaning + lemmatization), TF‑IDF feature extraction, classical ML (Logistic Regression, LightGBM) with hyperparameter search, and a feed‑forward neural network (TensorFlow/Keras). Visualizations (word clouds, coefficient bar chart) and evaluation reports are included.

Live demo : https://news-category-classification-zzmfsj23xhyrlthxynyet8.streamlit.app/

## Features
- Data cleaning and lemmatization using NLTK
- TF‑IDF feature extraction
- Model selection and hyperparameter tuning (Logistic Regression, LightGBM)
- Feed‑forward neural network with TF‑IDF input
- Word clouds and feature importance visualization
- Classification reports and evaluation utilities
- Streamlit UI demo for interactive sentiment analysis

## Project Structure

- `model.pkl` `vectorizer.pkl` - model and vectorizer files
- `News Category Classification.ipynb` - Jupyter Notebook containing preprocessing, training, and evaluation pipeline
- `app.py` - Streamlit application script for live sentiment analysis
- `requirements.txt` - Required Python packages
- `README.md` - Project overview and usage instructions

## Installation

1. Clone the repository:
``` powershell
git clone https://github.com/MohamedGamal04/News-Category-Classification.git
cd News-Category-Classification
```
2. Create a Python environment and activate it (recommended):
``` powershell
conda create -n classify-env python=3.8
conda activate classify-env
```
3. Install dependencies:
``` powershell
pip install -r requirements.txt
```
4. Download necessary NLTK data resources:
``` python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```
   
Run required NLTK downloads in the notebook (cells include downloads for punkt, stopwords, wordnet, averaged_perceptron_tagger).

## Usage

- Train the model and evaluate using the Jupyter Notebook.
- To use the Streamlit app (once created and set up), run:
``` terminal
streamlit run app.py
```

- The Streamlit UI allows you to enter reviews manually or upload CSV files for batch prediction.

## Dataset

This project uses the [AG news classification dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).

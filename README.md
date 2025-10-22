# News Category Classification
A complete end-to-end notebook for classifying news articles into four categories (World, Sports, Business, Sci/Tech) using the AG News dataset. The project demonstrates data loading, preprocessing (cleaning + lemmatization), TF‑IDF feature extraction, classical ML (Logistic Regression, LightGBM) with hyperparameter search, and a feed‑forward neural network (TensorFlow/Keras). Visualizations (word clouds, coefficient bar chart) and evaluation reports are included.

## Features
- Data cleaning and lemmatization using NLTK
- TF‑IDF feature extraction
- Model selection and hyperparameter tuning (Logistic Regression, LightGBM)
- Feed‑forward neural network with TF‑IDF input
- Word clouds and feature importance visualization
- Classification reports and evaluation utilities

## Requirements (recommended)
- Python 3.8+
- numpy, pandas, scikit-learn
- tensorflow (or tensorflow-cpu)
- lightgbm
- nltk, num2words, contractions
- matplotlib, plotly, wordcloud

## Installation (Windows PowerShell)
Create & activate virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Install packages:
```powershell
pip install -r requirements.txt
```
If no requirements.txt:
```powershell
pip install numpy pandas scikit-learn tensorflow lightgbm nltk num2words contractions matplotlib plotly wordcloud
```
   
Run required NLTK downloads in the notebook (cells include downloads for punkt, stopwords, wordnet, averaged_perceptron_tagger).

## Usage
- Open `News Category Classification.ipynb` in VS Code or Jupyter Notebook and run cells sequentially.
- The notebook:
  - downloads and loads the AG News dataset,
  - preprocesses text,
  - extracts TF‑IDF features,
  - runs classical models with GridSearchCV,
  - trains a Keras feed‑forward network using a TF‑IDF batch generator,
  - prints classification reports and visualizations.

## Acknowledgments
- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).

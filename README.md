Music Genre Classification using Machine Learning
This project classifies audio tracks into genres using feature extraction with Librosa and machine learning. The dataset is referenced from this Kaggle notebook.

Project Overview
The goal is to predict the genre of a music track based on audio features. The workflow includes:

Audio preprocessing and feature extraction

Label encoding and dataset preparation

Training a machine learning model for genre classification

Evaluating model performance using accuracy and confusion matrix

Project Structure
music-genre-classification.ipynb: Jupyter Notebook with the complete pipeline

dataset/: Directory expected to contain genre-labeled .wav audio files in subfolders 

Techniques Used
Feature Extraction: MFCC, chroma, spectral contrast using Librosa

Preprocessing: Normalization, label encoding, train-test split

Modeling: Random Forest, SVM, or a simple Neural Network

Evaluation: Accuracy score, classification report, confusion matrix

Requirements
This project requires the following Python libraries:

numpy

pandas

librosa

scikit-learn

matplotlib

seaborn

Make sure these are installed in your Python environment before running the notebook.

How to Use
Clone or download this repository

Download the dataset from the Kaggle source linked below and place it in a folder named dataset, organized by genre (e.g., dataset/rock, dataset/jazz, etc.)

Open the Jupyter Notebook music-genre-classification.ipynb

Run each cell to process the data, train the model, and evaluate its performance

Results
The notebook produces:

Evaluation metrics like accuracy, precision, recall, and F1-score

Confusion matrix and classification report

Visualization of model performance

Model accuracy may vary based on dataset size, balance, and choice of features.

Notes
Ensure your dataset is balanced across genres to reduce bias

Experimenting with model types and hyperparameters can improve results

Spectrogram-based CNN models may perform better on larger datasets

Dataset Reference
The dataset used in this project is available at:
https://www.kaggle.com/code/dapy15/music-genre-classification

Future Improvements
Implement convolutional neural networks using spectrograms

Create a web-based interface for user-uploaded genre prediction

Enable real-time classification from microphone input

References
https://www.tensorflow.org/datasets/catalog/gtzan

https://www.kaggle.com/code/dapy15/music-genre-classification

https://www.clairvoyant.ai/blog/music-genre-classification-using-cnn

https://github.com/alikaratana/Music-Genre-Classification



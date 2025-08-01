Music Genre Classification using Machine Learning
This project focuses on classifying music into different genres using machine learning. It involves audio data preprocessing, feature extraction using Librosa, and training a classifier to predict the genre of given audio samples.

Project Structure
music-genre-classification.ipynb: Main Jupyter Notebook

dataset/: Folder containing audio files (not included)

Each genre should be stored in a separate subfolder with .wav files

Techniques Used
Feature Extraction: MFCC, chroma, spectral contrast using Librosa

Data Preprocessing: Normalization, label encoding, train-test split

Model Training: Machine learning classifiers such as Random Forest, SVM, or Neural Network

Evaluation: Accuracy score, classification report, and confusion matrix

Requirements
Python libraries required for this project include:

numpy

pandas

librosa

scikit-learn

matplotlib

seaborn

These libraries must be installed in your Python environment before running the notebook.

How to Run
Clone or download this repository.

Place your dataset in a folder named dataset, structured by genre subfolders.

Open the notebook using Jupyter Notebook or Jupyter Lab.

Run the notebook step by step, ensuring the dataset path matches your folder.

Results
The notebook outputs performance metrics such as accuracy, a classification report, and visualizations like the confusion matrix. Model performance depends on the quality and balance of the dataset.

Notes
A well-balanced and labeled dataset is important for accurate results.

You can improve the model by experimenting with more features or different classifiers.

Spectrogram-based deep learning models can also be implemented for potentially better performance.

References
Librosa Documentation: https://librosa.org/

GTZAN Genre Dataset: http://marsyas.info/downloads/datasets.html

Future Work
Incorporate deep learning methods (e.g., CNN on spectrogram images)

Build a web-based interface for real-time genre prediction

Enable real-time audio classification from microphone input

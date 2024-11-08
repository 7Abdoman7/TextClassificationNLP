from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from datasets import load_dataset

# Prepare the dataset as before
import numpy as np

# Load the Stanford Sentiment Treebank (SST) dataset
dataset = load_dataset("stanfordnlp/sst", trust_remote_code=True)

# Prepare the dataset: We'll use the training set for training and test set for evaluation
trainSentences = dataset['train']['sentence']
trainClasses = dataset['train']['label']
testSentences = dataset['test']['sentence']
testClasses = dataset['test']['label']

# Define the thresholds for binning into classes
thresholds = [0.2, 0.4, 0.6, 0.8]

# Digitize the values into classes based on the thresholds
trainClasses = np.digitize(trainClasses, thresholds)
testClasses = np.digitize(testClasses, thresholds)

# Use scikit-learn to reproduce results
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
pipeline.fit(trainSentences, trainClasses)

# Predict on the test set
predictedClasses = pipeline.predict(testSentences)

# Calculate accuracy
accuracy = accuracy_score(testClasses, predictedClasses)
print(f"Accuracy for Naive Bayes with scikit-learn: {accuracy * 100:.2f}%")

# Confusion Matrix
confMatrix = confusion_matrix(testClasses, predictedClasses)
print("Confusion Matrix:")
print(confMatrix)

# Calculate precision, recall, and F1 score
precision = precision_score(testClasses, predictedClasses, average=None)
recall = recall_score(testClasses, predictedClasses, average=None)
f1 = f1_score(testClasses, predictedClasses, average=None)

macro_precision = precision_score(testClasses, predictedClasses, average='macro')
macro_recall = recall_score(testClasses, predictedClasses, average='macro')
macro_f1 = f1_score(testClasses, predictedClasses, average='macro')

print("Per-class Precision:", precision)
print("Per-class Recall:", recall)
print("Per-class F1 Score:", f1)
print("Macro-averaged Precision:", macro_precision)
print("Macro-averaged Recall:", macro_recall)
print("Macro-averaged F1 Score:", macro_f1)

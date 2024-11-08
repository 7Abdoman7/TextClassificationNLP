import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load sentences from dataset
sentences = set()
with open('./TextClassificationNLP/Assignment1/datasetSentences.txt', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        parts = line.split('\t')
        if len(parts) == 2:
            sentences.add(parts[1].strip())

uniqueSentences = list(sentences)

# Load sentiment labels
allLabels = []
with open('./TextClassificationNLP/Assignment1/sentiment_labels.txt', 'r') as file:
    next(file)
    for line in file:
        allLabels.append(float(line.split('|')[1]))

# Digitize the labels into classes
classes = np.array(allLabels[:len(uniqueSentences)])
classes = np.digitize(classes, [0.2, 0.4, 0.6, 0.8, 1.0], right=True)

# Shuffle and split the data into training and testing sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    uniqueSentences, classes, test_size=0.1, random_state=42)

# Function to generate bi-grams from a list of words
def generate_bigrams(words):
    return [(words[i], words[i + 1]) for i in range(len(words) - 1)]

# Step 1: Tokenize sentences and generate bi-grams for each sentence
all_bigrams = []
sentence_bigrams = []
for sentence in uniqueSentences:
    words = sentence.split()
    bigrams = generate_bigrams(words)
    sentence_bigrams.append(bigrams)
    all_bigrams.extend(bigrams)

# Step 2: Identify unique bi-grams and assign each a unique index
unique_bigrams = list(set(all_bigrams))
bigram_to_index = {bigram: idx for idx, bigram in enumerate(unique_bigrams)}

# Step 3: Initialize an empty sparse feature matrix
num_sentences = len(uniqueSentences)
num_bigrams = len(unique_bigrams)
feature_matrix = np.zeros((num_sentences, num_bigrams), dtype=np.int8)

# Step 4: Populate the feature matrix with 1s for existing bi-grams
for i, bigrams in enumerate(sentence_bigrams):
    for bigram in bigrams:
        if bigram in bigram_to_index:
            feature_matrix[i, bigram_to_index[bigram]] = 1

# Convert train and test data to numpy arrays
train_features = feature_matrix[:len(train_sentences)]
test_features = feature_matrix[len(train_sentences):]

# Initialize and train a logistic regression model using sklearn
model = LogisticRegression(max_iter=300, solver='lbfgs', multi_class='multinomial')
model.fit(train_features, train_labels)

# Predict on test data
test_predictions = model.predict(test_features)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(test_labels, test_predictions)
conf_matrix = confusion_matrix(test_labels, test_predictions)

print(f"Test Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

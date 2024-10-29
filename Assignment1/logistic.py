import torch
import numpy as np

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
combined = list(zip(uniqueSentences, classes))
np.random.shuffle(combined)
uniqueSentences, classes = zip(*combined)

trainSize = int(0.90 * len(uniqueSentences))

trainSentences = uniqueSentences[:trainSize]
testSentences = uniqueSentences[trainSize:]

trainClasses = classes[:trainSize] 
testClasses = classes[trainSize:]

trainDataDocument = dict(zip(trainSentences, trainClasses))
testDataDocument = dict(zip(testSentences, testClasses))

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

# Convert feature matrix and labels to torch tensors with minimal size and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_matrix = torch.tensor(feature_matrix, dtype=torch.int8, device=device)
classes = torch.tensor(classes, dtype=torch.int64, device=device)

# Initialize parameters
def initialize_parameters(num_features, num_classes):
    W = torch.randn((num_features, num_classes), dtype=torch.float32, device=device) * 0.01
    b = torch.zeros((1, num_classes), dtype=torch.float32, device=device)
    return W, b

# Softmax function
def softmax(z):
    exp_z = torch.exp(z - torch.max(z, dim=1, keepdim=True).values)
    return exp_z / torch.sum(exp_z, dim=1, keepdim=True)

# Cross-entropy loss
def compute_loss(Y, Y_hat):
    m = Y.shape[0]
    log_likelihood = -torch.log(Y_hat[torch.arange(m), Y])
    return torch.sum(log_likelihood) / m

# Forward propagation
def forward_propagation(X, W, b):
    z = torch.mm(X.to(torch.float32), W) + b  # Converting X to float32 for multiplication
    return softmax(z)

# Backward propagation
def backward_propagation(X, Y, Y_hat):
    m = X.shape[0]
    grad_W = torch.mm(X.to(torch.float32).T, (Y_hat - Y)) / m
    grad_b = torch.sum(Y_hat - Y, dim=0, keepdim=True) / m
    return grad_W, grad_b

# Update parameters
def update_parameters(W, b, grad_W, grad_b, learning_rate):
    W -= learning_rate * grad_W
    b -= learning_rate * grad_b
    return W, b

# Convert labels to one-hot encoded matrix
def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros((labels.size(0), num_classes), device=device)
    one_hot[torch.arange(labels.size(0)), labels] = 1
    return one_hot

# Modify train_model to save the best model parameters
def train_model(X, Y, X_dev, Y_dev, num_classes, learning_rate=0.01, epochs=100, batch_size=32):
    num_samples, num_features = X.shape
    W, b = initialize_parameters(num_features, num_classes)
    Y_one_hot = one_hot_encode(Y, num_classes)

    best_accuracy = float('-inf')
    best_params = None
    early_stop_count = 0

    for epoch in range(epochs):
        # Shuffle the dataset at the start of each epoch
        indices = torch.randperm(num_samples, device=device)
        X_shuffled = X[indices]
        Y_shuffled = Y_one_hot[indices]

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            # Forward propagation
            Y_hat = forward_propagation(X_batch, W, b)
            
            # Compute gradients
            grad_W, grad_b = backward_propagation(X_batch, Y_batch, Y_hat)
            del X_batch, Y_batch, Y_hat
            
            # Update parameters
            W, b = update_parameters(W, b, grad_W, grad_b, learning_rate)
            del grad_W, grad_b

        # Compute loss every epoch for tracking
        Y_hat_full = forward_propagation(X, W, b)
        loss = compute_loss(Y, Y_hat_full)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        del Y_hat_full  # Remove unused tensor after each epoch
        torch.cuda.empty_cache()
        
        test_predictions = predict(X_dev, W, b)
        accuracy = calculate_accuracy(Y_dev, test_predictions)
        print(f"Epoch {epoch+1}/{epochs}, accuracy: {accuracy:.4f}")

        # Save the parameters if current accuracy is bigger than best_accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'W': W.clone(), 'b': b.clone()}
            torch.save(best_params, 'best_model.pth')
            print(f"Best model saved at epoch {epoch+1} with accuracy {best_accuracy:.4f}")
        else:
            early_stop_count += 1
            
        if early_stop_count > 10:
            print(f"Early stopping at epoch {epoch+1} with accuracy {best_accuracy:.4f}")
            break

    return W, b

# Prediction function
def predict(X, W, b):
    Y_hat = forward_propagation(X, W, b)
    return torch.argmax(Y_hat, dim=1)

# Calculate accuracy
def calculate_accuracy(true_labels, predictions):
    correct_predictions = torch.sum(true_labels == predictions)
    return (correct_predictions / len(true_labels)).item() * 100

# Calculate confusion matrix
def calculate_confusion_matrix(true_labels, predictions, num_classes):
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32, device=device)
    for true, pred in zip(true_labels, predictions):
        confusion_matrix[true, pred] += 1
    return confusion_matrix

# Example usage with features and labels
num_classes = 5  # Example: number of sentiment classes
learning_rate = 0.001
epochs = 300
batch_size = 1024

test_feature_matrix = feature_matrix[trainSize:]
test_true_labels = classes[trainSize:]

# Train the model
W, b = train_model(feature_matrix, classes, test_feature_matrix, test_true_labels, num_classes, learning_rate, epochs, batch_size)

# Predict on test data 
test_predictions = predict(test_feature_matrix, W, b)
checkpoint = torch.load('best_model.pth', weights_only=True)
W, b = checkpoint['W'], checkpoint['b']

# Calculate accuracy and confusion matrix
accuracy = calculate_accuracy(test_true_labels, test_predictions)
confusion_matrix = calculate_confusion_matrix(test_true_labels, test_predictions, num_classes)

print(f"Test Accuracy: {accuracy:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix.cpu().numpy())
    
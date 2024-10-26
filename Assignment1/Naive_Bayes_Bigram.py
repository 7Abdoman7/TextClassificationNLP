import numpy as np

# Load sentences from dataset
sentences = set()
with open(r'C:\Users\Abdo\OneDrive\Desktop\NLP\Assignments\Assignment1\datasetSentences.txt', 'r', encoding='utf-8') as file:
    next(file) 
    for line in file:
        parts = line.split('\t')
        if len(parts) == 2:
            sentences.add(parts[1].strip())

uniqueSentences = list(sentences)

# Load sentiment labels
allLabels = []
with open(r"C:\Users\Abdo\OneDrive\Desktop\NLP\Assignments\Assignment1\sentiment_labels.txt", 'r') as file:
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

trainSize = int(0.95 * len(uniqueSentences))

trainSentences = uniqueSentences[:trainSize]
testSentences = uniqueSentences[trainSize:]

trainClasses = classes[:trainSize] 
testClasses = classes[trainSize:]

trainDataDocument = dict(zip(trainSentences, trainClasses))
testDataDocument = dict(zip(testSentences, testClasses))

# Helper function to generate bigrams from a sentence
def getBigrams(sentence):
    words = sentence.split()
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    return bigrams

# Train the Naive Bayes classifier with bigrams
def trainNaiveBayes(documents, targetClass):
    logLikelihood = {}
    count = 0

    numberOfDocuments = len(documents)
    numberOfDocumentsInClass = sum(1 for documentClass in documents.values() if documentClass == targetClass)

    logPrior = np.log(numberOfDocumentsInClass / numberOfDocuments)

    vocab = set()
    for sentence in documents.keys():
        bigrams = getBigrams(sentence)
        vocab.update(bigrams)
    vocabList = list(vocab)
    vocabLength = len(vocabList)

    bigDoc = []
    numberOfBigramsInClass = 0
    for sentence, label in documents.items():
        if label == targetClass:
            bigrams = getBigrams(sentence)
            bigDoc.extend(bigrams)
            numberOfBigramsInClass += len(bigrams)

    for bigram in vocabList:
        count = bigDoc.count(bigram)
        logLikelihood[bigram] = np.log((count + 1) / (numberOfBigramsInClass + vocabLength))  # Laplace smoothing

    return logPrior, logLikelihood, vocabList

# Test the Naive Bayes classifier
def testNaiveBayes(testSentence, logPriors, logLikelihoods, classes, vocabList):
    logPosteriors = np.zeros(len(classes))

    testBigrams = getBigrams(testSentence)

    for class_ in classes:
        logPosteriors[class_] += logPriors[class_]

        for testBigram in testBigrams:
            if testBigram in vocabList:
                logPosteriors[class_] += logLikelihoods[class_].get(testBigram, np.log(1 / (len(vocabList) + 1)))
            else:
                logPosteriors[class_] += np.log(1 / (len(vocabList) + 1))

    return np.argmax(logPosteriors)

# Calculate accuracy of the Naive Bayes classifier
def calculateNaiveBayesAccuracy(testDataDocument, logPriors, logLikelihoods, classes, vocabList):
    correctPredictions = 0
    predictedDataDocument = {}
    for sentence, actualClass in testDataDocument.items():
        predictedClass = testNaiveBayes(sentence, logPriors, logLikelihoods, classes, vocabList)
        predictedDataDocument[sentence] = predictedClass
        if predictedClass == actualClass:
            correctPredictions += 1

    accuracy = correctPredictions / len(testDataDocument) * 100
    return accuracy, predictedDataDocument

# Calculate the confusion matrix
def calculateConfusionMatrix(testDataDocument, predictedDataDocument, classes):
    confusionMatrix = np.zeros((len(classes), len(classes)), dtype=int)

    for testSentence, actualClass in testDataDocument.items():
        predictedClass = predictedDataDocument[testSentence]
        confusionMatrix[actualClass][predictedClass] += 1

    return confusionMatrix

# Define classes and initialize log priors and log likelihoods
classes = np.array([0, 1, 2, 3, 4])
logPriors = np.zeros(len(classes))
logLikelihoods = [{} for _ in classes]
vocabList = []

# Train Naive Bayes model for each class
for class_ in classes:
    logPriors[class_], logLikelihoods[class_], vocabList = trainNaiveBayes(trainDataDocument, class_)

# Calculate accuracy and predicted document
accuracy, predictedDataDocument = calculateNaiveBayesAccuracy(testDataDocument, logPriors, logLikelihoods, classes, vocabList)
print(f"Accuracy for Naive Bayes: {accuracy:.2f}%")

# Calculate and display confusion matrix
confusionMatrix = calculateConfusionMatrix(testDataDocument, predictedDataDocument, classes)
print("Confusion Matrix:")
print(confusionMatrix)

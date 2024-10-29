import numpy as np

sentences = set()
with open('./TextClassificationNLP/Assignment1/datasetSentences.txt', 'r', encoding='utf-8') as file:
    next(file) 
    for line in file:
        parts = line.split('\t')
        if len(parts) == 2:
            sentences.add(parts[1].strip())

uniqueSentences = list(sentences)

allLabels = []
with open('./TextClassificationNLP/Assignment1/sentiment_labels.txt', 'r') as file:
    next(file) 
    for line in file:
        allLabels.append(float(line.split('|')[1]))


classes = np.array(allLabels[:len(uniqueSentences)])
classes = np.digitize(classes, [0.2, 0.4, 0.6, 0.8, 1.0], right=True)


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

def trainNaiveBayes(documents, targetClass):
    logLikelihood = {}
    count = 0

    numberOfDocuments = len(documents)
    numberOfDocumentsInClass = sum(1 for documentClass in documents.values() if documentClass == targetClass)
    print(numberOfDocumentsInClass)
    logPrior = np.log(numberOfDocumentsInClass / numberOfDocuments)

    vocab = set()
    for sentence in documents.keys():
        words = sentence.split()  
        vocab.update(words)
    vocabList = list(vocab)
    vocabLength = len(vocabList)

    bigDoc = []
    numberOfWordsInClass = 0
    for sentence, label in documents.items():  
        if label == targetClass:
            words = sentence.split()
            bigDoc.extend(words)  
            numberOfWordsInClass += len(words) 

    for word in vocabList:
        count = bigDoc.count(word)  
        logLikelihood[word] = np.log((count + 1) / (numberOfWordsInClass + vocabLength))

    return logPrior, logLikelihood, vocabList


def testNaiveBayes(testSentence, logPriors, logLikelihoods, classes, vocabList):
    logPosteriors = np.zeros(len(classes))

    testWords = testSentence.split()

    for class_ in classes:
        logPosteriors[class_] += logPriors[class_]  

        for testWord in testWords:
            if testWord in vocabList:  
                logPosteriors[class_] += logLikelihoods[class_][testWord]
            else:  
                logPosteriors[class_] += np.log(1 / (len(vocabList) + 1))

    return np.argmax(logPosteriors)

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
    

def calculateConfusionMatrix(testDataDocument, predictedDataDocument, classes):
    confusionMatrix = np.zeros((len(classes), len(classes)), dtype=int)
    for testSentence, actualClass in testDataDocument.items():
        predictedClass = predictedDataDocument[testSentence]
        confusionMatrix[actualClass][predictedClass] += 1
    return confusionMatrix

def calculateMetrics(confusionMatrix):
    numClasses = confusionMatrix.shape[0]
    precision = np.zeros(numClasses)
    recall = np.zeros(numClasses)
    f1_score = np.zeros(numClasses)

    for i in range(numClasses):
        truePositive = confusionMatrix[i, i]
        falsePositive = confusionMatrix[:, i].sum() - truePositive
        falseNegative = confusionMatrix[i, :].sum() - truePositive

        precision[i] = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0
        recall[i] = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) > 0 else 0
        f1_score[i] = (
            2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1_score = f1_score.mean()

    return precision, recall, f1_score, macro_precision, macro_recall, macro_f1_score


classes = np.array([0, 1, 2, 3, 4])
logPriors = np.zeros(len(classes))
logLikelihoods = [{}, {}, {}, {}, {}]
vocabList = []

for class_ in classes:
    logPriors[class_], logLikelihoods[class_], vocabList = trainNaiveBayes(trainDataDocument, class_)
 
accuracy, predictedDataDocument = calculateNaiveBayesAccuracy(testDataDocument, logPriors, logLikelihoods, classes, vocabList)
print(f"Accuracy for Naive Bayes: {accuracy:.2f}%")

confusionMatrix = calculateConfusionMatrix(testDataDocument, predictedDataDocument, classes)
precision, recall, f1_score, macro_precision, macro_recall, macro_f1_score = calculateMetrics(confusionMatrix)

print("Per-class Precision:", precision)
print("Per-class Recall:", recall)
print("Per-class F1 Score:", f1_score)
print("Macro-averaged Precision:", macro_precision)
print("Macro-averaged Recall:", macro_recall)
print("Macro-averaged F1 Score:", macro_f1_score)
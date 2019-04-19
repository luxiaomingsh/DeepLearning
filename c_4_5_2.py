import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import math
import operator


def loadDataset():
    df = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    X = df.iloc[0:150, [0, 2]].values
    plt.scatter(X[0:50, 0], X[:50, 1], color='blue',
                marker='x', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='red',
                marker='o', label='versicolor')
    plt.scatter(X[100:150, 0], X[100:150, 1], color='green',
                marker='*', label='virginica')
    plt.xlabel('petal width')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def EuclidDist(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainSet)):
        dist = EuclidDist(testInstance, trainSet[x], length)
        distances.append((trainSet[x], dist))
        distances.sort(key=lambda distances: distances[1])
        neighbors = []
    for x in range(k):
        neighbors .append(distances[x][0])
    return neighbors


def getClass(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        instance_class = neighbors[x][-1]
        if instance_class in classVotes:
            classVotes[instance_class] += 1
    else:
        classVotes[instance_class] = 1
        sortedVotes = sorted(classVotes.items(),
                             key=operator.itemgetter(1), reverse=True)

    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    trainingSet = []
    testSet = []
    split = 0.7
    loadDataset('/Users/luxiaoming/VsProjrcts/ts1/dnn3/iris.data',
                split, trainingSet, testSet)
    print ('ilX:' + repr(len(trainingSet)))
    print ('iU:' + repr(len(testSet)))
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getClass(neighbors)
        predictions . append(result)
        print('> fiJ=' + repr(result) + ', YI=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('OtX:' + repr(accuracy) + '%')


main()

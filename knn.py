# K-nearest Neighbor classifier in Python.
# CoAuthor: Cody

import numpy
import random
import time
from scipy.spatial import distance

testingFiles = ["iris-testing.txt", "iris-pc-testing.txt"]
trainingFiles = ["iris-training.txt", "iris-pc-training.txt"]

for fileIndex in range(2):
  trainingFile = trainingFiles[fileIndex]
  testingFile = testingFiles[fileIndex]

  print(trainingFiles[fileIndex])

  x_train = numpy.loadtxt(trainingFile)

  numTrainingLines = x_train.shape[0]
  numTrainingColumns = x_train.shape[1]

  x_test = numpy.loadtxt(testingFile)
  numTestingLines = x_test.shape[0]

  listOfKValues = [5, 11, 21]
  for k in listOfKValues:
    # Positively identified counts
    true_positive_count = 0
    true_negative_count = 0
    # Falsely identified counts
    false_positive_count = 0 
    false_negative_count = 0

    for testLine in x_test:
      label1Count = 0
      labelnegative1Count = 0

      testLineLabel = testLine[numTrainingColumns - 1]
      distances = []
      for line in x_train:
        distances.append([
          distance.euclidean(line[0:numTrainingColumns - 2], testLine[0:numTrainingColumns - 2]),
          line[numTrainingColumns - 1]
        ])
      distances.sort()
      for i in range(0, k):
        trainingLabel = distances[k][1]

      if trainingLabel == 1:
        label1Count += 1
      elif trainingLabel == -1:
        labelnegative1Count += 1

      assumedCategory = 1
      if (labelnegative1Count > label1Count):
        assumedCategory = -1

      if (assumedCategory == 1):
        if (testLineLabel == 1):
          true_positive_count += 1
        else:
          false_positive_count += 1
      elif assumedCategory == -1:
        if (testLineLabel == -1):
          true_negative_count += 1
        else:
          false_negative_count += 1

    print('--- ---')
    print('k =', k)
    accuracy = (true_positive_count + true_negative_count) / numTestingLines
    print('accuracy: ', accuracy)
    # portion of testing data positives that were correcly identified
    sensitivity = true_positive_count / (true_positive_count + false_negative_count)
    print('sensitivity: ', sensitivity)
    # portion of testing data negatives that were correctly identifified
    specificity = true_negative_count / (false_positive_count + true_negative_count)
    print('specificity: ', specificity)
    precision = true_positive_count / (true_positive_count + false_positive_count)
    print('precision: ', precision)
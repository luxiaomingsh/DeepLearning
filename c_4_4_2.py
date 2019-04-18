from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        headings = next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def train_test_split(dataset, percent):
    train = list()
    train_size = percent * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy .pop(index))
    return train, dataset_copy


def mean(values):
    return sum(values) / float(len(values))


def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i]-mean_x)*(y[i]-mean_y)
    return covar


# Calculate the variance of a list of numbers
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])


def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    w1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    w0 = y_mean-w1*x_mean

    plt.axis([0, 200, 0, 600])
    plt.plot(x, y, 'go')
    plt.grid()
    plt.show()
    return (w0, w1)


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


def simple_linear_regression(train, test):
    predictions = list()
    w0, w1 = coefficients(train)
    for row in test:
        y_model = w1*row[0]+w0
        predictions . append(y_model)
    return predictions


def evaluate_algorithm(dataset, algorithm, split_percent, *args):
    train, test = train_test_split(dataset, split_percent)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set. append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse


seed(2)
filename = '/Users/luxiaoming/VsProjrcts/ts1/dnn3/insurance.csv'
dataset = load_csv(filename)
for col in range(len(dataset[0])):
    str_column_to_float(dataset, col)

percent = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression, percent)
print('RMSE: %.3f' % (rmse))

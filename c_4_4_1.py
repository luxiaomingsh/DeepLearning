import matplotlib.pyplot as plt
import numpy as np

dataset = [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]

# junzhi
def mean(values):
    return sum(values)/float(len(values))

# fangcha
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

# xiefangcha
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i]-mean_x)*(y[i]-mean_y)
    return covar

# huiguixishu
def coefficients(dataset):
    x_mean, y_mean = mean(x), mean(y)
    w1 = covariance(x, x_mean, y, y_mean)/variance(x, x_mean)
    w0 = y_mean-w1*x_mean
    return w0, w1

#huiguixishu
def simple_linear_regression(train,test):
    predict=list()
    w0, w1 = coefficients(train)
    for row in test:
        y_model=w1*row[0]+w0
        predict.append(y_model)
    return predict

#junfanggenwucha
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return np.sqrt(mean_error)


def evaluate_algorithm(dataset, algorithm):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
        predicted = algorithm(dataset, test_set)
    for val in predicted:
        print('%.3f\t' % (val))

    actual = [row[-1] for row in dataset]
    rmse = rmse_metric(actual, predicted)
    return rmse


rmse = evaluate_algorithm(dataset, simple_linear_regression)
print('RMSE: %.3f' % (rmse))



plt.axis([0, 6, 0, 6])
plt.plot(x, y, 'go')
plt.grid()
# plt.show()

mean_x, mean_y = mean(x), mean(y)
var_x, var_y = variance(x, mean_x), variance(y, mean_y)

print('%.3f ,%.3f' % (mean_x, var_x))
print('%.3f ,%.3f' % (mean_y, var_y))

w0, w1 = coefficients(dataset)
print('%.3f ,%.3f' % (w0, w1))

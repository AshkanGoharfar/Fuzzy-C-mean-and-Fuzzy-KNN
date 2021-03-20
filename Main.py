from plot_inputs import plot_inputs
from FCM_Operations import *
from FKNN_Operations import fknn
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

'''
    Initial attributes
'''

# FILENAME = 'data_set/sample1.csv'
FILENAME = 'data_set/sample2.csv'
# FILENAME = 'data_set/sample3.csv'
# FILENAME = 'data_set/sample4.csv'
# FILENAME = 'data_set/sample5.csv'
# M = 5
RADUIS = .8
m = 2
'''
    Start with plotting inputs
'''
plot_inputs(FILENAME)
INPUT = []
Y = []
MAX_C = 20
best_c = 4
FCM_M = 2
# Fuzzy KNN parameter
FKNN_K = 3
FKNN_M = 2


def read_data(filename):
    num_of_column = 0
    num_of_row = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row != []:
                num_of_row += 1
                num_of_column = len(row)
                input_row = []
                for i in range(num_of_column):
                    input_row.append(float(row[i]))
                INPUT.append(input_row)
                Y.append(0)
    csv_file.close()
    return num_of_column, num_of_row


if __name__ == '__main__':
    num_of_column, num_of_row = read_data(FILENAME)
    z = list(zip(np.array(INPUT), np.array(Y)))
    random.shuffle(z)
    INPUT, Y = zip(*z)
    len_INPUT = int(3 / 4 * len(INPUT)) - 1
    len_Y = int(3 / 4 * len(Y)) - 1
    x_train = INPUT[0:len_INPUT]
    x_test = INPUT[len_INPUT + 1:len(INPUT) - 1]
    y_train = Y[0:len_Y]
    y_test = Y[len_Y:len(Y) - 1]

    all_c = []
    all_entropy = []
    for c in range(1, MAX_C):
        center, entropy, u_best = fcm(np.array(INPUT), x_train, y_train, x_test, y_test, FCM_M, c)
        all_c.append(c)
        # all_entropy.append(entropy * np.power(c - best_c, 2))
        all_entropy.append(entropy)
        calculate_entropy(u_best, np.array(INPUT), center)
    plt.plot(all_c, all_entropy)
    plt.show()
    centers, entropy, u_best = fcm(np.array(INPUT), x_train, y_train, x_test, y_test, FCM_M, best_c)
    fknn(np.array(INPUT), u_best, FKNN_K, best_c, FKNN_M)

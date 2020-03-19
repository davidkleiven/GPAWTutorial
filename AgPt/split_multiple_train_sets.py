import numpy as np
from random import shuffle

data = np.loadtxt('agpt_train.csv', delimiter=',')
folder = "/work/sophus/nestedlasso"
num_sets = 100
dataset_size = 200

with open('agpt_train.csv', 'r') as infile:
    header = infile.readline()
    header = header.replace('# ', '')
    header = header.strip()


indices = list(range(data.shape[0]))
for i in range(num_sets):
    shuffle(indices)
    dataset = data[indices[:dataset_size], :]
    np.savetxt(folder + "/dataset{}.csv".format(i), dataset, header=header, delimiter=',')

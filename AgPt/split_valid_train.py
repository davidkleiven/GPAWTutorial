import numpy as np
from random import shuffle

num_valid = 300
orig_fname = "agpt_data.csv"

data = np.loadtxt(orig_fname, delimiter=',')
indices = list(range(data.shape[0]))
shuffle(indices)
indices = np.array(indices)

with open(orig_fname, 'r') as infile:
    header = infile.readline()
    header = header.replace('# ', '')
    header = header.strip()

validate = data[indices[:num_valid], :]
train = data[indices[num_valid:], :]

print(len(header.split(',')))
np.savetxt('agpt_train.csv', train, delimiter=',', header=header)
np.savetxt('agpt_validate.csv', validate, delimiter=',', header=header)

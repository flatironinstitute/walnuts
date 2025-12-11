import numpy as np
import json

data = np.loadtxt("german.data-numeric", dtype=float)
x = data[:, :-1]
y = (data[:, -1] - 1).astype(int)
N, P = x.shape
data_dict = {'N': N, 'P': P, 'x': x.tolist(), 'y': y.tolist()}
with open('german-credit-data.json', 'w') as json_file:
    json.dump(data_dict, json_file, indent=2)    

import numpy as np
import os

path_file = os.getcwd()
data = np.load(os.path.join(path_file,"data/tardata.npy"))


print(data[20])



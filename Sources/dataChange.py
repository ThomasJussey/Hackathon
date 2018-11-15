import pickle
import numpy as np
from PIL import Image
import os
from tempfile import TemporaryFile

Xdata = pickle.load( open( "../Data/data_40000/X_train.p", "rb" ))
Ydata = pickle.load( open( "../Data/data_40000/y_train.p", "rb" ))

doublons = pickle.load( open("../Data/doublon.p", "rb" ))

print(len(Xdata))
new_doublons= np.delete(doublons, 1, 1)  # delete second column of C
new_Ydata = np.delete(Ydata, new_doublons)

#on créé d une liste qui va contenir tous les indices des couples a retenir
d = []
for i in range(40000):
    if i not in new_doublons:
        d.append(i)

new_Xdata = Xdata[d]
print(len(new_Xdata))


with open("../Data/data_39212/X_train.p", 'wb') as handle:
    pickle.dump(new_Xdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("../Data/data_39212/y_train.p", 'wb') as handle:
    pickle.dump(new_Ydata, handle, protocol=pickle.HIGHEST_PROTOCOL)

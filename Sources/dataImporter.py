import pickle
import numpy as np
from PIL import Image
import os
import time
debut = time.time()

Xdata = pickle.load( open( "../Data/data_2000/X_train.p", "rb" ))


#data[0] = couple d'image 1
#data[0][0] = image 1 du couple 1
#data[0][1] = image 2 du couple 1
#data[0][0][31] = colonne 32 de l'image
#data[0][0][0][0] = Premier pixel

#my_array = data[0][1].reshape((32, 32)).astype('uint8') #*255 pour avoir le négatif
#im = Image.fromarray(my_array)
#im.save("../Data/images/result.jpg")

i = 0
j = 0
for i in range(0,2000):
    if not os.path.exists("../Data/images/%d" % (i)):
        os.makedirs("../Data/images/%d" % (i))
        for j in range(0,2):
                my_array = Xdata[i][j].reshape((32, 32)).astype('uint8')
                im = Image.fromarray(my_array)
                im.save("../Data/images/%d/result%d%d.jpg" % (i,i,j))
    else:
        print("This directory already exists")



Ydata = pickle.load( open( "../Data/data_2000/Y_train.p", "rb" ))


y_array = []
for i in range(0,2000):
    y_array.append(Ydata[i])
    print(y_array[i] == 1)

np.savetxt("../Data/images/results.txt", y_array)


fin = time.time()
print("Temps : %d sec" %(fin - debut))

from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

x_train, x_test, y_train, y_test= model_selection.train_test_split(x, y, train_size=0.3)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.preprocessing.image import img_to_array
from tqdm import tqdm
np.random.seed(42)
size = 256

clean_data = []
path1 = 'E:/Code/MRI_de_aliasing/data/BraTS2020_T1/test_GT2/'
files = os.listdir(path1)
for i in tqdm(files):
    img = cv2.imread(path1+'/' + i, 0) #0 for gray image
    img =  cv2.resize(img, (size, size,))
    clean_data.append(img_to_array(img))

noisy_data = []
path2 = 'E:/Code/MRI_de_aliasing/data/BraTS2020_T1/test_input/UP_202/'

files = os.listdir(path2)
for i in tqdm(files):
    img = cv2.imread(path2+'/' + i, 0) #0 for gray image
    img =  cv2.resize(img, (size, size))
    noisy_data.append(img_to_array(img))


print('clean data shape', len(clean_data))
print('noisy data shape', len(noisy_data))
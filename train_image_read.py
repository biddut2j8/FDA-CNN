import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.preprocessing.image import img_to_array
from tqdm import tqdm

np.random.seed(42)
size = 256

clean_data = []
path1 = 'E:/Code/MRI_de_aliasing/data/BraTS2020_T1/train_GT/'
files = os.listdir(path1)
for i in tqdm(files):
    img = cv2.imread(path1+'/' + i, 0) #0 for gray image
    img =  cv2.resize(img, (size, size,))
    clean_data.append(img_to_array(img))

noisy_data = []
path2 = 'E:/Code/MRI_de_aliasing/data/BraTS2020_T1/train_input/UP_20/'

files = os.listdir(path2)
for i in tqdm(files):
    img = cv2.imread(path2+'/' + i, 0) #0 for gray image
    img =  cv2.resize(img, (size, size))
    noisy_data.append(img_to_array(img))

clean_train = np.reshape(clean_data, (len(clean_data), size, size, 1))
clean_train = clean_train.astype('float32') /255.

noisy_train = np.reshape(noisy_data, (len(noisy_data), size, size, 1))
noisy_train = noisy_train.astype('float32') /255.

print('clean data shape', clean_train.shape)
print('noisy data shape', noisy_train.shape)

def image_display2(image_f, image_u):
    for slice_position in range(image_f.shape[0]):
        plt.subplot(221),plt.imshow((image_f[slice_position]), cmap = 'gray')
        plt.title('Clean image'), plt.xticks([]), plt.yticks([])

        plt.subplot(222),plt.imshow((image_u[slice_position]), cmap = 'gray')
        plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])
        if(slice_position>=10):
            break
        plt.show()

#image_display2(clean_train, noisy_train)

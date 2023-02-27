import glob
import numpy as np
import argparse
import h5py
import tensorflow as tf
from matplotlib import pyplot as plt

#E:\Code\Python\MRI_de_aliasing\kspace\Brats_T1\test

parser = argparse.ArgumentParser(description='Train dealiasing Model')
parser.add_argument('--imshape', default=(256,256), type=int, help='imshape')
parser.add_argument('--train_data', default='../kspace/IXI_T2/test2/*.h5', type=str, help='path of train data')
parser.add_argument('--num_channel', default=2, type=int, help='num_channel')
parser.add_argument('--batch_size', default= 8, type=int, help='batch_size')
parser.add_argument('--epoch', default= 100, type=int, help='number_of_epochs')
parser.add_argument('--mask_20_random', default='../mask/gussian_mask_2D_25_random.npy', type=str, help='path of random 20 mask')
parser.add_argument('--mask_25_random', default='../mask/gussian_mask_2D_60_random.npy', type=str, help='path of random 25 mask')
parser.add_argument('--mask_20_uniform', default='../mask/gussian_mask_1D_50_row.npy', type=str, help='path of uniform 20 mask')
parser.add_argument('--mask_25_uniform', default='../mask/gussian_mask_1D_75_row.npy', type=str, help='path of uniform 25 mask')
args = parser.parse_args()

def slice_number(filename):
    data_accept = len(filename)
    data_file = h5py.File(filename[1], 'r')
    kspace_read = data_file['kspace']
    slice_per_k = kspace_read.shape[0]
    total_sample = data_accept * slice_per_k
    return total_sample

def full_k_read(filename):
    kspace_full = np.zeros((slice_number(filename),args.imshape[0],args.imshape[1], args.num_channel))
    slice_counter = 0
    for i in range(len(filename)):
        data_file = h5py.File(filename[i], 'r')
        kspace_read = data_file['kspace']
        slice_per_kspace = kspace_read.shape[0]
        kspace_full[slice_counter:slice_counter+slice_per_kspace,:,:,0] = kspace_read[:,:,:].real
        kspace_full[slice_counter:slice_counter+slice_per_kspace,:,:,1] = kspace_read[:,:,:].imag
        slice_counter+=slice_per_kspace
    return kspace_full

def random_under_sampled(filename, mask):
    under_k = np.array(filename)
    under_k[:, mask, :] = 0
    return under_k

def make_complex_k(ksapce_real_imaginary):
    real = ksapce_real_imaginary[:, :, :, 0]
    imag = ksapce_real_imaginary[:, :, :, 1]
    k_complex = tf.complex(real, imag)
    return k_complex

def ifft_shift_layer(kspace):
    reconstruct_image = (tf.abs(tf.signal.fftshift(tf.signal.ifft2d(kspace))))
    return reconstruct_image

def mask_display(uniform, random, af):
    plt.figure()
    plt.subplot(121)
    plt.imshow(uniform, )
    plt.axis("off")
    plt.title("uniform sampled:{}%".format(af))

    plt.subplot(122)
    plt.imshow(random, )
    plt.axis("off")
    plt.title("uniform sampled:{}%".format(af))
    plt.show()

def kspace_image_display(full_k, under_k):
    for slice_position in range(full_k.shape[0]):
        full_ks = tf.math.log(tf.abs(full_k[slice_position]))
        plt.subplot(221),plt.imshow(full_ks,cmap = 'gray' )
        plt.title('Ground Truth--k'), plt.xticks([]), plt.yticks([])

        full_i = ifft_shift_layer(full_k)
        plt.subplot(222),plt.imshow((full_i[slice_position]), cmap = 'gray' )
        plt.title('Ground Truth--i'), plt.xticks([]), plt.yticks([])

        under_ks = tf.math.log(tf.abs(under_k[slice_position]))
        plt.subplot(223),plt.imshow(under_ks, cmap = 'gray' )
        plt.title('under-k'), plt.xticks([]), plt.yticks([])

        under_i = ifft_shift_layer(under_k)
        plt.subplot(224),plt.imshow((under_i[slice_position]),cmap = 'gray')
        plt.title('under-i'), plt.xticks([]), plt.yticks([])
        if (slice_position >= 10):
            break
        print(slice_position)
        plt.show()

def kspace_display(full_k, under_k):
    for slice_position in range(full_k.shape[0]):
        full_ks = tf.math.log(tf.abs(full_k[slice_position]))
        plt.subplot(121),plt.imshow(full_ks, )
        plt.title(''), plt.xticks([]), plt.yticks([])

        under_ks = tf.math.log(tf.abs(under_k[slice_position]))
        plt.subplot(122),plt.imshow(under_ks, )
        plt.title(''), plt.xticks([]), plt.yticks([])

        if (slice_position >= 100):
            break
        plt.show()

def image_display(image_f, image_u):
    for slice_position in range(image_f.shape[0]):
        plt.subplot(121), plt.imshow((image_f[slice_position]), cmap = 'gray' )
        plt.title('Ground Truth'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow((image_u[slice_position]), cmap = 'gray')
        plt.title('Under sampled', plt.xticks([]), plt.yticks([]))
        if (slice_position >= 10):
            break
        plt.show()

def image_save(full_image, under_image, location= 'uni_fixed'):
    slice_position = 0
    for i in range((full_image.shape[0])):
        image = (full_image[i])
        #plt.imsave('E:/Code/Python/MRI_de_aliasing/data/FastMRI_T1/reconstruct/original/{}.jpg'.format(slice_position), image, cmap = 'gray')

        image2 = (under_image[i])
        plt.imsave('../data/IXI_T2/unet_reconstruct/{}/{}.jpg'.format(location, slice_position), image2, cmap = 'gray')
        slice_position = slice_position+1
    print('successfully saved {} images into {}'.format(under_image.shape[0], location))

#undersampling Read mask
random_20_mask = np.load(args.mask_20_random)
random_25_mask = np.load(args.mask_25_random)

uniform_20_mask = np.load(args.mask_20_uniform)
uniform_25_mask = np.load(args.mask_25_uniform)

#mask_display(random_20_mask, random_25_mask, af= 25)

#data read
kspace_train_files = (glob.glob(args.train_data))
full_k = full_k_read(kspace_train_files)
print('fk shape', full_k.shape)
#print('f k mean, std', np.mean(full_k), np.std(full_k))

#random Sampled
#uk = random_under_sampled(full_k, uniform_20_mask)
# print('fk shape', uk.shape)
#print('u k mean, std', np.mean(uk), np.std(uk))

#fixed uniform data
import read_h5 as d5
uk = d5.under_k_read_25_percent(kspace_train_files)

#make complex data
complex_f_k = (make_complex_k(full_k))
full_k= 0
complex_u_k = (make_complex_k(uk))
uk = 0
#kspace_display(complex_f_k, complex_u_k)

#generate image
complex_f_k = ifft_shift_layer(complex_f_k)
print('combine shape', complex_f_k.shape)

complex_u_k = ifft_shift_layer(complex_u_k)
print('combine u k shape', complex_u_k.shape)

#image_display(complex_f_k, complex_u_k)
image_save(complex_f_k, complex_u_k)
#kspace_image_display(complex_f_k, complex_u_k)


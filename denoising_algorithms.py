
'''import glob
import skimage
import numpy as np
import cv2
from skimage import img_as_float, io
from matplotlib import pyplot as plt
from scipy import ndimage as nd
from skimage.metrics import peak_signal_noise_ratio
import time
from skimage.color import rgb2gray
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma, cycle_spin)
'''
'''image_number =1
clean_test = (img_as_float(io.imread("E:/Code/Python/MRI_de_aliasing/data/IXI_T2/reconstruct/original/{}.jpg".format(image_number), as_gray=True)))
#plt.imsave('E:/Code/Python/My_MRI_model/image/denoising_algorithms/original_image.jpg', ref_img, cmap= 'gray')
#noisy_img= skimage.util.random_noise(ref_img, mode="gaussian")
#plt.imsave('E:/Code/Python/My_MRI_model/image/denoising_algorithms/noisy_img.jpg', noisy_img, cmap= 'gray')

noisy_test = (img_as_float(io.imread("E:/Code/Python/MRI_de_aliasing/data/IXI_T2/reconstruct/uni_random/{}.jpg", as_gray=True)))
'''
import time
import math
from sklearn.metrics import mean_squared_error
from skimage.metrics import normalized_root_mse as nrmse, structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse
from sewar import full_ref

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.preprocessing.image import img_to_array
from tqdm import tqdm
np.random.seed(42)
size = 256

#E:\BIDDUT\Python\MRI_de_aliasing\data\BraTS2020_T1\test\original


clean_data = []
path1 = 'E:/Code/FDA_CNN (paper)/data/IXI_T2/reconstruct/original'
files = os.listdir(path1)
for i in tqdm(files):
    img = cv2.imread(path1+'/' + i, 0) #0 for gray image
    img =  cv2.resize(img, (size, size,))
    clean_data.append(img_to_array(img))

noisy_data = []
path2 = 'E:/Code/FDA_CNN (paper)/data/IXI_T2/reconstruct/random'

files = os.listdir(path2)
for i in tqdm(files):
    img = cv2.imread(path2+'/' + i, 0) #0 for gray image
    img =  cv2.resize(img, (size, size))
    noisy_data.append(img_to_array(img))


print('clean data shape', len(clean_data))
print('noisy data shape', len(noisy_data))

def PSNR(original, predictions):
    mse = np.mean((original - predictions) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    Psnr = 10 * math.log10((pixel_max / math.sqrt(mse)))
    return Psnr

def NRMSE(original, prediction):
    rms = math.sqrt(mean_squared_error(original, prediction))
    Nrmse = rms/(original.max() - prediction.min())
    return Nrmse


clean_test = np.reshape(clean_data, (len(clean_data), size, size))
clean_test = clean_test.astype('float32')/255.

noisy_test = np.reshape(noisy_data, (len(noisy_data), size, size))
noisy_test = noisy_test.astype('float32')/255.

number_files = (noisy_test.shape[0])
met = np.zeros([4, number_files])

method = 'Reconstruct'
start = int(round(time.time()))




for slice_position in range(clean_test.shape[0]):

    ref_img = clean_test[slice_position]
    restore_image = noisy_test[slice_position]

    ssim_score = ssim(ref_img, restore_image, channel_axis=True)
    psnr_skimg = PSNR(ref_img, restore_image)
    rmse_skimg = NRMSE(ref_img, restore_image)
    VIFP_img = full_ref.vifp(ref_img, restore_image, sigma_nsq=2)

    met[0, slice_position] = ssim_score
    met[1, slice_position] = psnr_skimg
    met[2, slice_position] = rmse_skimg
    met[3, slice_position] = VIFP_img

    slice_position = slice_position + 1

    # save data
    # plt.imsave('E:/Code/MRI_de_aliasing/data/BraTS2020_T1/test_output/UP_20/{}/{}.jpg'.format(method, slice_position), restore_image, cmap='gray')
    #np.save('E:/BIDDUT/Python/MRI_de_aliasing/data/fastMRI_T1/output_metrics/random/met_{}.npy'.format(method), met)

    # display
    plt.subplot(121), plt.imshow(ref_img, cmap = 'gray' )
    plt.title('Fully Sampled '), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(restore_image, cmap = 'gray')
    plt.title('Reconstruct image')
    '''plt.show()
    if slice_position>= 20:
         break'''

end = int(round(time.time()))
recons_time = end - start
undersampled = 'random'
dataset = 'IXI-T2'

print('\nMethod : {}'.format(method))
print('Input : {} {}'.format(dataset,undersampled))

print("Result: mean+/-std ")
print("ssim: %.2f +/- %.2f" % (met[0, :].mean(), met[0, :].std()))
print("psnr: %.2f +/- %.2f" % (met[1, :].mean(), met[1, :].std()))
print("nrmse: %.2f +/- %.2f" % (met[2, :].mean(), met[2, :].std()))
print("VIF: %.2f +/- %.2f " % (met[3, :].mean(), met[3, :].std()))
print('Reconstruction time: %.2f' %(recons_time/ clean_test.shape[0]))

print("\nmaximum ssim: %.2f and slice number %.0f" % (np.max(met[0]), np.argmax(met[0])))
print("maximum  psnr : %.2f and slice number %.0f" % (np.max(met[1]), np.argmax(met[1])))
print("minimum  nrmse: %.2f and slice number %.0f" % (np.min(met[2]), np.argmin(met[2])))
print("maximum  vif: %.2f and slice number %.0f" % (np.max(met[3]), np.argmax(met[3])))

# ################# gaussian method
# #noise_free_image= nd.gaussian_filter(noisy_img, sigma=5)
#
# ################# Bilateral
# # sigma_est = estimate_sigma(noisy_img, multichannel=True, average_sigmas=True)
# # noise_free_image = denoise_bilateral(noisy_img, sigma_spatial=15,multichannel=False)
#
#
# #### TV #################
# #noise_free_image = denoise_tv_chambolle(noisy_img, weight=0.3, multichannel=False)
#
# ####Wavelet #################
# #noise_free_image= denoise_wavelet(noisy_img, multichannel=False,method='BayesShrink', mode='soft',rescale_sigma=True)
#
#
# ################# Shift invariant wavelet denoising
# # denoise_kwargs = dict(multichannel=False, wavelet='db1', method='BayesShrink',rescale_sigma=True)
# # max_shifts = 3     #0, 1, 3, 5
# # noise_free_image = cycle_spin(noisy_img,func=denoise_wavelet,max_shifts =max_shifts,func_kw=denoise_kwargs,multichannel=False)
#
# ############## Anisotropic Diffusion
# # from medpy.filter.smoothing import anisotropic_diffusion
# # noise_free_image = anisotropic_diffusion(noisy_img, niter=50, kappa=50, gamma=0.2, option=2)
#
# ################################  non local means (NLM) from SKIMAGE
# from skimage.restoration import denoise_nl_means
# # sigma_est = np.mean(estimate_sigma(noisy_img, multichannel=False))
# # noise_free_image = denoise_nl_means(noisy_img, h=1.15 * sigma_est, fast_mode=True, patch_size=9, patch_distance=5, multichannel=False)
#
# ####################################################
# #MRF
# # Code from following github. It works but too slow and not as good as the above filters.
# #https://github.com/ychemli/Image-denoising-with-MRF/blob/master/ICM_denoising.py
# #Very slow... and not so great
#
# #BM3D Block-matching and 3D filtering (better)
# '''import bm3d
# noise_free_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
#
# start = time.time()'''
#
#
# #plt.imsave("E:/Code/Python/My_MRI_model/image/denoising_algorithms/MRF.jpg", noise_free_image, cmap='gray')
#
# '''noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
# gaussian_cleaned_psnr = peak_signal_noise_ratio(ref_img, noise_free_image)
# print("PSNR of input noisy image = ", noise_psnr)
# print("PSNR of cleaned image = ", gaussian_cleaned_psnr)
# end = time.time()
# print(f"Runtime of the W-net for one k-space is: {end - start}")
#
# plt.figure()
# plt.subplot(221)
# plt.imshow(ref_img,cmap = 'gray')
# plt.title('original image')
#
# plt.subplot(222)
# plt.imshow(noisy_img,cmap = 'gray')
# plt.title('noisy image')
#
# plt.subplot(223)
# plt.imshow(noise_free_image,cmap = 'gray')
# plt.title('noisy image')
#
# plt.show()'''
#
# ####make noise image of several noises
# '''
# def plotnoise(img, mode, r, c, i):
#     plt.subplot(r,c,i)
#     if mode is not None:
#         gimg = skimage.util.random_noise(img, mode=mode)
#         plt.imshow(gimg)
#     else:
#         plt.imshow(img)
#     plt.title(mode)
#     plt.axis("off")
#
# plt.figure(figsize=(18,24))
# r=4
# c=2
# plotnoise(ref_img, "gaussian", r,c,1)
# plotnoise(ref_img, "localvar", r,c,2)
# plotnoise(ref_img, "poisson", r,c,3)
# plotnoise(ref_img, "salt", r,c,4)
# plotnoise(ref_img, "pepper", r,c,5)
# plotnoise(ref_img, "s&p", r,c,6)
# plotnoise(ref_img, "speckle", r,c,7)
# plotnoise(ref_img, None, r,c,8)
# plt.show()
# '''
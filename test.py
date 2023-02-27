from keras.models import load_model
import numpy as np
import time
import matplotlib.pylab as plt
import os
from test_image_read import clean_data, noisy_data, size
from skimage.metrics import normalized_root_mse as nrmse, structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse
from sewar import full_ref


source_dir = "./AttentDenseUnet/2022-07-21_1001_250/"
model_name = "AttentDenseUnet"
model = load_model(os.path.join(source_dir,model_name+".h5"))
print("Loaded model from disk",model_name)
method = 'unet'

clean_test = np.reshape(clean_data, (len(clean_data), size, size))
clean_test = clean_test.astype('float32') /255.

noisy_test = np.reshape(noisy_data, (len(noisy_data), size, size))
noisy_test = noisy_test.astype('float32') /255.

loss, acc= model.evaluate(noisy_test, clean_test)
print('accuracy: {:5.2f}%'.format(100*acc))

img_test = noisy_test.reshape((noisy_test.shape[0],noisy_test.shape[1],noisy_test.shape[2], 1))

print('clean data shape', clean_test.shape)
print('noisy data shape', noisy_test.shape)

number_files = (noisy_test.shape[0])
met = np.zeros([5, number_files])


start = int(round(time.time()))
remove_noise = model.predict(img_test)
end = int(round(time.time()))
recons_time = end - start

print('model output shape', remove_noise.shape)
remove_noise = remove_noise.reshape((clean_test.shape[0], clean_test.shape[1], clean_test.shape[2]))
print('retore shape', remove_noise.shape)



for slice_position in range(clean_test.shape[0]):

    ref_img = clean_test[slice_position]
    noisy_test1 = noisy_test[slice_position]
    restore_image = remove_noise[slice_position]

    ssim_score = ssim(ref_img, restore_image, multichannel=True)
    psnr_skimg = psnr(ref_img, restore_image, data_range=ref_img.max() - ref_img.min())
    rmse_skimg = nrmse(ref_img, restore_image)
    UQI_img = full_ref.uqi(ref_img, restore_image, ws=8)
    VIFP_img = full_ref.vifp(ref_img, restore_image, sigma_nsq=2)

    met[0, slice_position] = ssim_score
    met[1, slice_position] = psnr_skimg
    met[2, slice_position] = rmse_skimg
    met[3, slice_position] = UQI_img
    met[4, slice_position] = VIFP_img

    slice_position = slice_position + 1

    # save data
    # plt.imsave('E:/Code/MRI_de_aliasing/data/BraTS2020_T1/test_output/UP_20/{}/{}.jpg'.format(method, slice_position), restore_image, cmap='gray')
    # np.save('E:/Code/MRI_de_aliasing/data/BraTS2020_T1/test_output/UP_20/{}/met_{}.npy'.format(method, method), met)

    # display
    plt.subplot(321), plt.imshow(ref_img, 'gray')
    plt.title('Ground image'), plt.xticks([]), plt.yticks([])
    plt.subplot(322), plt.imshow(noisy_test1, 'gray')
    plt.title("noisy image")
    plt.subplot(323), plt.imshow(restore_image, 'gray')
    plt.title('restore image')
    # plt.show()
    # if slice_position>= 10:
    #     break

print('\nmodel: {} {}'.format(source_dir,model_name))
print("average ssim, psnr, nrmse, mse, time ")
print("ssim: %.2f +/- %.2f" % (met[0, :].mean(), met[0, :].std()))
print("psnr: %.2f +/- %.2f" % (met[1, :].mean(), met[1, :].std()))
print("nrmse: %.2f +/- %.2f" % (met[2, :].mean(), met[2, :].std()))
print("UQI: %.2f +/- %.2f " % (met[3, :].mean(), met[3, :].std()))
print("VIF: %.2f +/- %.2f " % (met[4, :].mean(), met[4, :].std()))
print('Reconstruction time: %.2f' %(recons_time/ img_test.shape[0]))

print("\nmaximum ssim: %.2f and slice number %.0f" % (np.max(met[0]), np.argmax(met[0])))
print("maximum  psnr : %.2f and slice number %.0f" % (np.max(met[1]), np.argmax(met[1])))
print("minimum  nrmse: %.2f and slice number %.0f" % (np.min(met[2]), np.argmin(met[2])))
print("maximum  uqi: %.2f and slice number %.0f" % (np.max(met[3]), np.argmax(met[3])))
print("maximum  vif: %.2f and slice number %.0f" % (np.max(met[4]), np.argmax(met[4])))



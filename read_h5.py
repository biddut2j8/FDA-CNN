import numpy as np
import h5py
from config import args


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

#take 30 lines
def under_k_read_50_percent(filename):
    row_corner_s =0
    row_corner_f = 32

    row_middle_s = 45
    row_middle_f = 85

    row_center_s = 95
    row_center_f = 150

    under_kspace_train = np.zeros((slice_number(filename),args.imshape[0],args.imshape[1], args.num_channel))
    slice_counter = 0
    for i in range(len(filename)):
        data_file = h5py.File(filename[i], 'r')
        kspace_read = data_file['kspace']
        slice_per_kspace = kspace_read.shape[0]
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_corner_s:row_corner_f, :, 0] = kspace_read[:, row_corner_s:row_corner_f, :].real
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_corner_s:row_corner_f, :, 1] = (kspace_read[:, row_corner_s:row_corner_f, :].imag)

        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_middle_s:row_middle_f, :, 0] = (kspace_read[:, row_middle_s:row_middle_f, :].real)
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_middle_s:row_middle_f, :, 1] = (kspace_read[:, row_middle_s:row_middle_f, :].imag)

        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_center_s:row_center_f, :, 0] = (kspace_read[:, row_center_s:row_center_f, :].real)
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_center_s:row_center_f, :, 1] = (kspace_read[:, row_center_s:row_center_f, :].imag)
        slice_counter+=slice_per_kspace
    return under_kspace_train

#take 50 lines
def under_k_read_20_percent(filename):
    row_corner_s =0
    row_corner_f = 100

    row_middle_s = 101
    row_middle_f = 150

    row_center_s = 151
    row_center_f = 200

    under_kspace_train = np.zeros((slice_number(filename),args.imshape[0],args.imshape[1], args.num_channel))
    slice_counter = 0
    for i in range(len(filename)):
        data_file = h5py.File(filename[i], 'r')
        kspace_read = data_file['kspace']
        slice_per_kspace = kspace_read.shape[0]
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_corner_s:row_corner_f, :, 0] = kspace_read[:, row_corner_s:row_corner_f, :].real
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_corner_s:row_corner_f, :, 1] = (kspace_read[:, row_corner_s:row_corner_f, :].imag)

        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_middle_s:row_middle_f, :, 0] = (kspace_read[:, row_middle_s:row_middle_f, :].real)
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_middle_s:row_middle_f, :, 1] = (kspace_read[:, row_middle_s:row_middle_f, :].imag)

        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_center_s:row_center_f, :, 0] = (kspace_read[:, row_center_s:row_center_f, :].real)
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_center_s:row_center_f, :, 1] = (kspace_read[:, row_center_s:row_center_f, :].imag)
        slice_counter+=slice_per_kspace
    return under_kspace_train

#take 64 lines
def under_k_read_25_percent(filename):
    row_corner_s =0
    row_corner_f = 60

    row_middle_s = 70
    row_middle_f = 110

    row_center_s = 120
    row_center_f = 150

    under_kspace_train = np.zeros((slice_number(filename),args.imshape[0],args.imshape[1], args.num_channel))
    slice_counter = 0
    for i in range(len(filename)):
        data_file = h5py.File(filename[i], 'r')
        kspace_read = data_file['kspace']
        slice_per_kspace = kspace_read.shape[0]
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_corner_s:row_corner_f, :, 0] = kspace_read[:, row_corner_s:row_corner_f, :].real
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_corner_s:row_corner_f, :, 1] = (kspace_read[:, row_corner_s:row_corner_f, :].imag)

        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_middle_s:row_middle_f, :, 0] = (kspace_read[:, row_middle_s:row_middle_f, :].real)
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_middle_s:row_middle_f, :, 1] = (kspace_read[:, row_middle_s:row_middle_f, :].imag)

        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_center_s:row_center_f, :, 0] = (kspace_read[:, row_center_s:row_center_f, :].real)
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_center_s:row_center_f, :, 1] = (kspace_read[:, row_center_s:row_center_f, :].imag)
        slice_counter+=slice_per_kspace
    return under_kspace_train

def full_k_read_knee(filename):
    slice_per_k = 30
    total_sample = len(filename) * slice_per_k
    print('number of train sample', total_sample)

    kspace_full = np.zeros((total_sample,args.imshape[0],args.imshape[1], args.num_channel))

    slice_counter = 0
    for i in range(len(filename)):
        data_file = h5py.File(filename[i], 'r')
        kspace_read = data_file['kspace']
        slice_per_kspace = slice_per_k
        kspace_full[slice_counter:slice_counter+slice_per_kspace,:,:,0] = (kspace_read[0:slice_per_k,0:256,0:256].real)
        kspace_full[slice_counter:slice_counter+slice_per_kspace,:,:,1] = (kspace_read[0:slice_per_k,0:256,0:256].imag)
        slice_counter+=slice_per_kspace
    return kspace_full

def under_k_knee(filename):
    row_corner_s =0
    row_corner_f = 30

    row_middle_s = 50
    row_middle_f =80

    row_center_s = 100
    row_center_f = 160

    slice_per_k = 30
    total_sample = len(filename) * slice_per_k
    print('number of train sample', total_sample)

    under_kspace_train = np.zeros((total_sample,args.imshape[0],args.imshape[1], args.num_channel))

    slice_counter = 0
    for i in range(len(filename)):
        data_file = h5py.File(filename[i], 'r')
        kspace_read = data_file['kspace']
        slice_per_kspace = slice_per_k
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_corner_s:row_corner_f, :, 0] = (kspace_read[0:slice_per_k, row_corner_s:row_corner_f, 0:256].real)
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_corner_s:row_corner_f, :, 1] = (kspace_read[0:slice_per_k, row_corner_s:row_corner_f, 0:256].imag)

        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_middle_s:row_middle_f, :, 0] = (kspace_read[0:slice_per_k, row_middle_s:row_middle_f, 0:256].real)
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_middle_s:row_middle_f, :, 1] = (kspace_read[0:slice_per_k, row_middle_s:row_middle_f, 0:256].imag)

        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_center_s:row_center_f, :, 0] = (kspace_read[0:slice_per_k, row_center_s:row_center_f, 0:256].real)
        under_kspace_train[slice_counter:slice_counter+slice_per_kspace,row_center_s:row_center_f, :, 1] = (kspace_read[0:slice_per_k, row_center_s:row_center_f, 0:256].imag)
        slice_counter+=slice_per_kspace

    return under_kspace_train


def full_k_read_cal(filename):
    data_file = np.load(filename[1], 'r')
    slice_per_k = data_file.shape[0]
    total_sample = len(filename) * slice_per_k

    kspace_full = np.zeros((total_sample,args.imshape[0],args.imshape[1], args.num_channel))
    norm = np.sqrt(args.imshape[0] * args.imshape[1])
    slice_counter = 0
    for i in range(len(filename)):
        normalize_kspace = np.load(filename[i])/norm
        slice_per_kspace = normalize_kspace.shape[0]
        kspace_full[slice_counter:slice_counter+slice_per_kspace,:,:,0] = normalize_kspace[:,:,:,0]
        kspace_full[slice_counter:slice_counter+slice_per_kspace,:,:,1] = normalize_kspace[:,:,:,1]
        slice_counter+=slice_per_kspace
    return kspace_full

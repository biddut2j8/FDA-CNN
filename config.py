import argparse

parser = argparse.ArgumentParser(description='Train denoising Model')
parser.add_argument('--imshape', default=(256,256), type=int, help='imshape')
parser.add_argument('--size', default=256, type=int, help='imshape h w')
parser.add_argument('--train_data', default='E:/data/IXI/T1/train/*.h5', type=str, help='path of train data')
#parser.add_argument('--val_data', default='E:/data/fastMRI/T1/val2/*.h5', type=str, help='path of val data')
#parser.add_argument('--test_data', default='E:/data/fastMRI/T1/test/*.h5', type=str, help='path of test data')
parser.add_argument('--num_channel', default=2, type=int, help='num_channel')
parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
parser.add_argument('--epoch', default=100, type=int, help='number_of_epochs')

parser.add_argument('--knee_train_data', default='E:/data/knee/train/*.h5', type=str, help='path of knee data_train')
parser.add_argument('--knee_val_data', default='E:/data/knee/val/*.h5', type=str, help='path of knee data_val')
parser.add_argument('--knee_test_data', default='E:/data/knee/test/*.h5', type=str, help='path of knee data_test')

parser.add_argument('--cal_train_data', default='E:/data/Calgary/train/*.npy', type=str, help='path of Calgary data_train')
parser.add_argument('--cal_val_data', default='E:/data/Calgary/val/*.npy', type=str, help='path of Calgary data_val')
parser.add_argument('--cal_test_data', default='E:/data/Calgary/test/*.npy', type=str, help='path of Calgary data_test')


parser.add_argument('--len_k_file', default=50, type=int, help='number of kspace files for coding time')
parser.add_argument('--len_k_file_test', default=10, type=int, help='number of kspace files for testing time')

args = parser.parse_args()
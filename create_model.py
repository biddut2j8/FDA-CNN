import tensorflow as tf
from tensorflow.keras import Model, layers
#****************model******************
#unet
#aesr--autoencoder-sr
#wnet
#DnCNN_DRL
#pbcunet
#ARDU

size = 256

def conv_block_2(x, filter_size, size, dropout, batch_norm=False):
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def conv_block_1(x, filter_size, size, dropout, batch_norm=False):
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv

input_shape= (size, size, 1)
inputs = layers.Input(input_shape, dtype=tf.float32)

def unet(dropout_rate=0.0, batch_norm=False):
    FILTER_NUM = 64  # number of basic filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_64 = conv_block_2(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    # DownRes 2 convolution + pooling
    conv_128 = conv_block_2(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    # DownRes 3 convolution + pooling
    conv_256 = conv_block_2(pool_128, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_256 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)

    # DownRes 4 convolution + pooling
    conv_512 = conv_block_2(pool_256, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_512 = layers.MaxPooling2D(pool_size=(2, 2))(conv_512)

    # DownRes 5 convolution + pooling
    conv_1024 = conv_block_2(pool_512, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up1 = layers.UpSampling2D(size=(2, 2))(conv_1024)
    concate1 = layers.concatenate([conv_512, up1], axis=-1)
    conv_up_512 = conv_block_2(concate1, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # up2
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_512)
    concate2 = layers.concatenate([conv_256, up2], axis=-1)
    conv_up_256 = conv_block_2(concate2, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # up3
    up3 = layers.UpSampling2D(size=(2, 2))(conv_up_256)
    concate3 = layers.concatenate([conv_128, up3], axis=-1)
    conv_up_128 = conv_block_2(concate3, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    # up4
    up4 = layers.UpSampling2D(size=(2, 2))(conv_up_128)
    concate4 = layers.concatenate([conv_64, up4], axis=-1)
    conv_up_64 = conv_block_2(concate4, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    out = layers.Conv2D(1, (1, 1), activation='relu')(conv_up_64)
    model = Model(inputs=inputs, outputs=out)
    return model

def autoencoder_sr(dropout_rate=0.0, batch_norm=False):
    FILTER_NUM = 64  # number of basic filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_64 = conv_block_2(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    # DownRes 2 convolution + pooling
    conv_128 = conv_block_2(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    # DownRes 3
    conv_256 = conv_block_1(pool_128, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up1 = layers.UpSampling2D(size=(2, 2))(conv_256)
    concate1 = layers.concatenate([conv_128, up1], axis=-1)
    conv_up_128 = conv_block_2(concate1, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    #up2
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_128)
    concate2 = layers.concatenate([conv_64, up2], axis=-1)
    conv_up_128 = conv_block_2(concate2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    out = layers.Conv2D(1, (1, 1), activation='relu')(conv_up_128)
    model = Model(inputs=inputs, outputs=out)
    return model

def wnet():

    kshape2 = 3

    conv9 = layers.Conv2D(48, kshape2, activation='relu', padding='same')(inputs)
    conv9 = layers.Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    conv9 = layers.Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv9)

    conv10 = layers.Conv2D(64, kshape2, activation='relu', padding='same')(pool4)
    conv10 = layers.Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    conv10 = layers.Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(conv10)

    conv11 = layers.Conv2D(128, kshape2, activation='relu', padding='same')(pool5)
    conv11 = layers.Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    conv11 = layers.Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    pool6 = layers.MaxPooling2D(pool_size=(2, 2))(conv11)

    conv12 = layers.Conv2D(256, kshape2, activation='relu', padding='same')(pool6)
    conv12 = layers.Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    conv12 = layers.Conv2D(256, kshape2, activation='relu', padding='same')(conv12)

    up4 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv12), conv11],axis=-1)
    conv13 = layers.Conv2D(128, kshape2, activation='relu', padding='same')(up4)
    conv13 = layers.Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    conv13 = layers.Conv2D(128, kshape2, activation='relu', padding='same')(conv13)

    up5 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv13), conv10],axis=-1)
    conv14 = layers.Conv2D(64, kshape2, activation='relu', padding='same')(up5)
    conv14 = layers.Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    conv14 = layers.Conv2D(64, kshape2, activation='relu', padding='same')(conv14)

    up6 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv14), conv9],axis=-1)
    conv15 = layers.Conv2D(48, kshape2, activation='relu', padding='same')(up6)
    conv15 = layers.Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    conv15 = layers.Conv2D(48, kshape2, activation='relu', padding='same')(conv15)

    out = layers.Conv2D(1, (1, 1), activation='linear')(conv15)
    model = Model(inputs = inputs, outputs = out)
    return model

def DnCNN_DRL():

    # 1st layer, Conv+relu
    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    # 15 layers, Conv+BN+relu
    for i in range(5):
        x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = layers.LeakyReLU()(x)
    # last layer, Conv
    x = layers.Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    out = layers.Subtract()([inputs, x])   # input - artifacts
    model = Model(inputs=inputs, outputs = out )

    return model

def pbcunet(dropout_rate=0.0, batch_norm=False):
    FILTER_NUM = 32  # number of basic filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter

#unet 1
    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_32 = conv_block_2(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)

    # DownRes 1, convolution + pooling
    conv_64 = conv_block_2(pool_32, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    # DownRes 2 convolution + pooling
    conv_128 = conv_block_2(pool_64, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    # DownRes 3 convolution + pooling
    conv_256 = conv_block_2(pool_128, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_256 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)

    # DownRes 4 convolution
    conv_512 = conv_block_2(pool_256, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up1 = layers.UpSampling2D(size=(2, 2))(conv_512)
    concate1 = layers.concatenate([conv_256, up1], axis=-1)
    conv_up_512 = conv_block_2(concate1, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # up2
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_512)
    concate2 = layers.concatenate([conv_128, up2], axis=-1)
    conv_up_256 = conv_block_2(concate2, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # up3
    up3 = layers.UpSampling2D(size=(2, 2))(conv_up_256)
    concate3 = layers.concatenate([conv_64, up3], axis=-1)
    conv_up_128 = conv_block_2(concate3, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    # up4
    up4 = layers.UpSampling2D(size=(2, 2))(conv_up_128)
    concate4 = layers.concatenate([conv_32, up4], axis=-1)
    conv_up_64 = conv_block_2(concate4, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    conv_up_16 = layers.Conv2D(16, (1, 1), activation='relu')(conv_up_64)
    out1 = layers.Conv2D(1, (1, 1), activation='relu')(conv_up_16)

    model = Model(inputs=inputs, outputs=out1)
    return model

from tensorflow.keras import backend as K

def gating_signal(input, out_size):
    x = layers.Conv2D(out_size, (1, 1),activation= 'relu', padding='same')(input)
    return x

def repeat_elem(tensor, rep):
     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    # result_bn = layers.BatchNormalization()(result)
    return result


def ardu(dropout_rate=0.0, batch_norm=False):
    FILTER_NUM = 64  # number of basic filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_64 = conv_block_2(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    conv_64 = conv_block_1(conv_64, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    # DownRes 2 convolution + pooling
    conv_128 = conv_block_2(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    conv_128 = conv_block_1(conv_128, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    # DownRes 3
    conv_256 = conv_block_2(pool_128, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    conv_256 = conv_block_1(conv_256, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_256 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)
    #bottleneck
    conv_512 = conv_block_2(pool_256, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    conv_512 = conv_block_1(conv_512, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    gating_4 = gating_signal(conv_512, 4 * FILTER_NUM)
    att_4 = attention_block(conv_256, gating_4, 4 * FILTER_NUM)
    up0 = layers.UpSampling2D(size=(2, 2))(conv_512)
    concate0 = layers.concatenate([att_4, up0], axis=-1)
    conv_up_256 = conv_block_2(concate0, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    conv_up_256 = conv_block_1(conv_up_256, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    gating_3 = gating_signal(conv_up_256, 2 * FILTER_NUM)
    att_3 = attention_block(conv_128, gating_3, 2 * FILTER_NUM)
    up1 = layers.UpSampling2D(size=(2, 2))(conv_up_256)
    concate1 = layers.concatenate([att_3, up1], axis=-1)
    conv_up_128 = conv_block_2(concate1, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    conv_up_128 = conv_block_1(conv_up_128, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    #up2
    gating_2 = gating_signal(conv_up_128,  FILTER_NUM)
    att_2 = attention_block(conv_64, gating_2,  FILTER_NUM)
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_128)
    concate2 = layers.concatenate([att_2, up2], axis=-1)
    conv_up_64 = conv_block_2(concate2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    conv_up_64 = conv_block_1(conv_up_64, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    out = layers.Conv2D(1, (1, 1), activation='relu')(conv_up_64)
    out = layers.Subtract()([out, inputs])
    model = Model(inputs=inputs, outputs=out)
    return model

def ARDunet(dropout_rate=0.0, batch_norm=False):
    FILTER_NUM = 64  # number of basic filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_64 = conv_block_2(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    # DownRes 2 convolution + pooling
    conv_128 = conv_block_2(pool_64, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    # DownRes 3 convolution + pooling
    conv_256 = conv_block_2(pool_128, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_256 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)

    # DownRes 4 convolution + pooling
    conv_512 = conv_block_2(pool_256, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_512 = layers.MaxPooling2D(pool_size=(2, 2))(conv_512)

    # DownRes 5 convolution + pooling
    conv_1024 = conv_block_2(pool_512, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    gating_1 = gating_signal(conv_1024, 8 * FILTER_NUM)
    att_1 = attention_block(conv_512, gating_1, 8 * FILTER_NUM)
    up1 = layers.UpSampling2D(size=(2, 2))(conv_1024)
    concate1 = layers.concatenate([att_1, up1], axis=-1)
    conv_up_512 = conv_block_2(concate1, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # up2
    gating_2 = gating_signal(conv_up_512, 4 * FILTER_NUM)
    att_2 = attention_block(conv_256, gating_2, 4 * FILTER_NUM)
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_512)
    concate2 = layers.concatenate([att_2, up2], axis=-1)
    conv_up_256 = conv_block_2(concate2, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # up3
    gating_3 = gating_signal(conv_up_256, 2 * FILTER_NUM)
    att_3 = attention_block(conv_128, gating_3, 2 * FILTER_NUM)
    up3 = layers.UpSampling2D(size=(2, 2))(conv_up_256)
    concate3 = layers.concatenate([att_3, up3], axis=-1)
    conv_up_128 = conv_block_2(concate3, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    # up4
    gating_4 = gating_signal(conv_up_128, FILTER_NUM)
    att_4 = attention_block(conv_64, gating_4, FILTER_NUM)
    up4 = layers.UpSampling2D(size=(2, 2))(conv_up_128)
    concate4 = layers.concatenate([att_4, up4], axis=-1)
    conv_up_64 = conv_block_2(concate4, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    output_layer = layers.Conv2D(1, (1, 1), activation='relu')(conv_up_64)
    out = layers.Subtract()([output_layer, inputs])
    model = Model(inputs=inputs, outputs=out)
    return model

#dense net
from keras.optimizers import *
from keras.models import *
from keras.layers import  *
def DenseBlock(channels,inputs):

    conv1_1 = Conv2D(channels, (1, 1),activation=None, padding='same')(inputs)
    conv1_1=BatchActivate(conv1_1)
    conv1_2 = Conv2D(channels//4, (3, 3), activation=None, padding='same')(conv1_1)
    conv1_2 = BatchActivate(conv1_2)

    conv2=concatenate([inputs,conv1_2])
    conv2_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv2)
    conv2_1 = BatchActivate(conv2_1)
    conv2_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv2_1)
    conv2_2 = BatchActivate(conv2_2)

    conv3 = concatenate([inputs, conv1_2,conv2_2])
    conv3_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv3)
    conv3_1 = BatchActivate(conv3_1)
    conv3_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv3_1)
    conv3_2 = BatchActivate(conv3_2)

    conv4 = concatenate([inputs, conv1_2, conv2_2,conv3_2])
    conv4_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv4)
    conv4_1 = BatchActivate(conv4_1)
    conv4_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv4_1)
    conv4_2 = BatchActivate(conv4_2)
    result=concatenate([inputs,conv1_2, conv2_2,conv3_2,conv4_2])
    return result

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def DenseUNet(input_size=(256, 256, 1), start_neurons=64, keep_prob=0.9,block_size=7,lr=1e-3):
    # input_shape = (size, size, 1)
    # inputs = layers.Input(input_shape, dtype=tf.float32)

    #inputs = Input(input_size)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(inputs)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 1, conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = DenseBlock(start_neurons * 2, pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = DenseBlock(start_neurons * 4, pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    convm = DenseBlock(start_neurons * 8, pool3)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, (1, 1), activation=None, padding="same")(uconv3)
    uconv3 = BatchActivate(uconv3)
    uconv3 = DenseBlock(start_neurons * 4, uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same")(uconv2)
    uconv2 = BatchActivate(uconv2)
    uconv2 = DenseBlock(start_neurons * 2, uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same")(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = DenseBlock(start_neurons * 1, uconv1)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('relu')(output_layer_noActi)
    out = layers.Add()([output_layer, inputs])
    model = Model(inputs=inputs, outputs=out)
    return model


def AttentDenseUNet():
    start_neurons = 32

    con1 = Conv2D(1, (3, 3), activation=None, padding="same")(inputs)
    conv1 = DenseBlock(start_neurons * 1, con1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = DenseBlock(start_neurons * 2, pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = DenseBlock(start_neurons * 4, pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = DenseBlock(start_neurons * 8, pool3)
    pool4 = MaxPooling2D((2, 2))(conv4)

    convm = DenseBlock(start_neurons * 16, pool4)

    gating_4 = gating_signal(convm, 8 * start_neurons)
    att_4 = attention_block(conv4, gating_4, 8 * start_neurons)

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, att_4])
    uconv3 = Conv2D(start_neurons * 8, (1, 1), activation=None, padding="same")(uconv3)
    uconv3 = DenseBlock(start_neurons * 8, uconv3)

    gating_3 = gating_signal(uconv3, 4 * start_neurons)
    att_3 = attention_block(conv3, gating_3, 4 * start_neurons)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, att_3])
    #uconv2 = Conv2D(start_neurons * 4, (1, 1), activation=None, padding="same")(uconv2)
    uconv2 = DenseBlock(start_neurons * 4, uconv2)

    gating_2 = gating_signal(uconv2, 2 * start_neurons)
    att_2 = attention_block(conv2, gating_2, 2 * start_neurons)

    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, att_2])
    #uconv1 = Conv2D(start_neurons * 2, (1, 1), activation='relu', padding="same")(uconv1)
    uconv1 = DenseBlock(start_neurons * 2, uconv1)

    gating_1 = gating_signal(uconv1, start_neurons)
    att_1 = attention_block(conv1, gating_1,  start_neurons)

    deconv0 = Conv2DTranspose(start_neurons, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = concatenate([deconv0, att_1])
    #uconv0 = Conv2D(start_neurons * 2, (1, 1), activation='relu', padding="same")(uconv0)
    uconv0 = DenseBlock(start_neurons * 2, uconv0)


    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv0)
    #output_layer = Activation('relu')(output_layer_noActi)
    out = layers.Add()([output_layer_noActi, inputs])
    model = Model(inputs=inputs, outputs=out)
    return model


if __name__ == "__main__":

    model = AttentDenseUNet()
    print(model.summary())


'''
#UNet 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_32 = conv_block_2(conv_up_64, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)

    # DownRes 1, convolution + pooling
    conv_64 = conv_block_2(pool_32, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    # DownRes 2 convolution + pooling
    conv_128 = conv_block_2(pool_64, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    # DownRes 3 convolution + pooling
    conv_256 = conv_block_2(pool_128, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_256 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)

    # DownRes 4 convolution
    conv_512 = conv_block_2(pool_256, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up1 = layers.UpSampling2D(size=(2, 2))(conv_512)
    concate1 = layers.concatenate([conv_256, up1], axis=-1)
    conv_up_512 = conv_block_2(concate1, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # up2
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_512)
    concate2 = layers.concatenate([conv_128, up2], axis=-1)
    conv_up_256 = conv_block_2(concate2, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # up3
    up3 = layers.UpSampling2D(size=(2, 2))(conv_up_256)
    concate3 = layers.concatenate([conv_64, up3], axis=-1)
    conv_up_128 = conv_block_2(concate3, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    # up4
    up4 = layers.UpSampling2D(size=(2, 2))(conv_up_128)
    concate4 = layers.concatenate([conv_32, up4], axis=-1)
    conv_up_64 = conv_block_2(concate4, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # conv_up_16 = layers.Conv2D(16, (1, 1), activation='relu')(conv_up_64)
    # out2 = layers.Conv2D(1, (1, 1), activation='relu')(conv_up_16)

#Unet 3
    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_32 = conv_block_2(conv_up_64, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)

    # DownRes 1, convolution + pooling
    conv_64 = conv_block_2(pool_32, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    # DownRes 2 convolution + pooling
    conv_128 = conv_block_2(pool_64, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    # DownRes 3 convolution + pooling
    conv_256 = conv_block_2(pool_128, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_256 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)

    # DownRes 4 convolution
    conv_512 = conv_block_2(pool_256, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up1 = layers.UpSampling2D(size=(2, 2))(conv_512)
    concate1 = layers.concatenate([conv_256, up1], axis=-1)
    conv_up_512 = conv_block_2(concate1, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # up2
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_512)
    concate2 = layers.concatenate([conv_128, up2], axis=-1)
    conv_up_256 = conv_block_2(concate2, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # up3
    up3 = layers.UpSampling2D(size=(2, 2))(conv_up_256)
    concate3 = layers.concatenate([conv_64, up3], axis=-1)
    conv_up_128 = conv_block_2(concate3, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    # up4
    up4 = layers.UpSampling2D(size=(2, 2))(conv_up_128)
    concate4 = layers.concatenate([conv_32, up4], axis=-1)
    conv_up_64 = conv_block_2(concate4, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # conv_up_16 = layers.Conv2D(16, (1, 1), activation='relu')(conv_up_64)
    # out3 = layers.Conv2D(1, (1, 1), activation='relu')(conv_up_16)

#Unet 4
    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_32 = conv_block_2(conv_up_64, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)

    # DownRes 1, convolution + pooling
    conv_64 = conv_block_2(pool_32, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    # DownRes 2 convolution + pooling
    conv_128 = conv_block_2(pool_64, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    # DownRes 3 convolution + pooling
    conv_256 = conv_block_2(pool_128, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_256 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)

    # DownRes 4 convolution
    conv_512 = conv_block_2(pool_256, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up1 = layers.UpSampling2D(size=(2, 2))(conv_512)
    concate1 = layers.concatenate([conv_256, up1], axis=-1)
    conv_up_512 = conv_block_2(concate1, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # up2
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_512)
    concate2 = layers.concatenate([conv_128, up2], axis=-1)
    conv_up_256 = conv_block_2(concate2, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # up3
    up3 = layers.UpSampling2D(size=(2, 2))(conv_up_256)
    concate3 = layers.concatenate([conv_64, up3], axis=-1)
    conv_up_128 = conv_block_2(concate3, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    # up4
    up4 = layers.UpSampling2D(size=(2, 2))(conv_up_128)
    concate4 = layers.concatenate([conv_32, up4], axis=-1)
    conv_up_64 = conv_block_2(concate4, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # conv_up_16 = layers.Conv2D(16, (1, 1), activation='relu')(conv_up_64)
    # out4 = layers.Conv2D(1, (1, 1), activation='relu')(conv_up_16)

#Unet 5
    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_32 = conv_block_2(conv_up_64, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)

    # DownRes 1, convolution + pooling
    conv_64 = conv_block_2(pool_32, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)

    # DownRes 2 convolution + pooling
    conv_128 = conv_block_2(pool_64, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_128 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)

    # DownRes 3 convolution + pooling
    conv_256 = conv_block_2(pool_128, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_256 = layers.MaxPooling2D(pool_size=(2, 2))(conv_256)

    # DownRes 4 convolution
    conv_512 = conv_block_2(pool_256, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    up1 = layers.UpSampling2D(size=(2, 2))(conv_512)
    concate1 = layers.concatenate([conv_256, up1], axis=-1)
    conv_up_512 = conv_block_2(concate1, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # up2
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_512)
    concate2 = layers.concatenate([conv_128, up2], axis=-1)
    conv_up_256 = conv_block_2(concate2, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # up3
    up3 = layers.UpSampling2D(size=(2, 2))(conv_up_256)
    concate3 = layers.concatenate([conv_64, up3], axis=-1)
    conv_up_128 = conv_block_2(concate3, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    # up4
    up4 = layers.UpSampling2D(size=(2, 2))(conv_up_128)
    concate4 = layers.concatenate([conv_32, up4], axis=-1)
    conv_up_64 = conv_block_2(concate4, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    conv_up_16 = layers.Conv2D(16, (1, 1), activation='relu')(conv_up_64)
    out = layers.Conv2D(1, (1, 1), activation='relu')(conv_up_16)

    model = Model(inputs=inputs, outputs=out)
    return model
'''


import tensorflow as tf
from tensorflow.keras import Model, layers

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
    #result_bn = layers.BatchNormalization()(result)
    return result

input_shape= (size, size, 1)
inputs = layers.Input(input_shape, dtype=tf.float32)

def RDUA(dropout_rate = 0.0, batch_norm = False):
    FILTER_NUM = 32  # number of basic filters for the first layer
    FILTER_SIZE = 3  # size of the convolutional filter

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_0 = conv_block_1(inputs, FILTER_SIZE, 48, dropout_rate, batch_norm)
    conv_1 = conv_block_2(conv_0, FILTER_SIZE, 48, dropout_rate, batch_norm)
    #conv_1 = conv_block_1(conv_1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(conv_1)

    # DownRes 2 convolution + pooling
    conv_2 = conv_block_1(pool_1, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    conv_2 = conv_block_2(conv_2, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

    # DownRes 3 convolution + pooling
    conv_3 = conv_block_1(pool_2, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    conv_3 = conv_block_2(conv_3, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    pool_3 = layers.MaxPooling2D(pool_size=(2, 2))(conv_3)

    # DownRes 4 convolution + pooling
    conv_4 = conv_block_1(pool_3, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    conv_4 = conv_block_2(conv_4, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    pool_512 = layers.MaxPooling2D(pool_size=(2, 2))(conv_4)

    # DownRes 5 convolution + pooling
    conv_1024 = conv_block_2(pool_512, FILTER_SIZE, 16 * FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    gating_4 = gating_signal(conv_1024, 8 * FILTER_NUM)
    att_4 = attention_block(conv_4, gating_4, 8 * FILTER_NUM)
    up1 = layers.UpSampling2D(size=(2, 2))(conv_1024)
    concate1 = layers.concatenate([att_4, up1], axis=-1)
    conv_up_512 = conv_block_2(concate1, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)
    conv_up_512 = conv_block_1(conv_up_512, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate, batch_norm)

    # up2
    gating_3 = gating_signal(conv_up_512, 4 * FILTER_NUM)
    att_3 = attention_block(conv_3, gating_3, 4 * FILTER_NUM)
    up3 = layers.UpSampling2D(size=(2, 2))(conv_4)
    concate3 = layers.concatenate([att_3, up3], axis=-1)
    conv_up_3= conv_block_2(concate3, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)
    conv_up_3 = conv_block_1(conv_up_3, FILTER_SIZE, 4 * FILTER_NUM, dropout_rate, batch_norm)

    # up3
    gating_2 = gating_signal(conv_up_3, 2 * FILTER_NUM)
    att_2 = attention_block(conv_2, gating_2, 2 * FILTER_NUM)
    up2 = layers.UpSampling2D(size=(2, 2))(conv_up_3)
    concate2 = layers.concatenate([att_2, up2], axis=-1)
    conv_up_2 = conv_block_2(concate2, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)
    conv_up_2 = conv_block_1(conv_up_2, FILTER_SIZE, 2 * FILTER_NUM, dropout_rate, batch_norm)

    # up4
    gating_1 = gating_signal(conv_up_2, FILTER_NUM)
    att_1 = attention_block(conv_1, gating_1, FILTER_NUM)
    up1 = layers.UpSampling2D(size=(2, 2))(conv_up_2)
    concate1 = layers.concatenate([att_1, up1], axis=-1)
    conv_up_1 = conv_block_2(concate1, FILTER_SIZE, 48, dropout_rate, batch_norm)
    conv_up_1 = conv_block_1(conv_up_1, FILTER_SIZE, 48, dropout_rate, batch_norm)


    #conv_up_0 = conv_block_2(conv_up_1, FILTER_SIZE, 64, dropout_rate, batch_norm)
    out = layers.Conv2D(1, (1, 1), activation='linear')(conv_up_1)

    out = layers.Add()([out, inputs])
    model = Model(inputs=inputs, outputs=out)
    return model

if __name__ == "__main__":
    model = RDUA()
    print(model.summary())
    #print(model.get_weights())

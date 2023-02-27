import pathlib, datetime, time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard
from matplotlib import pyplot
import create_model as bm_model
#import my_model_rdua as rdua
from sklearn.model_selection import train_test_split
import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For CuDnn
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

def allocate_gpu_memory(gpu_number=0):
    print(device_lib.list_local_devices())
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.set_visible_devices(physical_devices[gpu_number], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
            print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")

'''dataAugmentaion = ImageDataGenerator(rotation_range = 30, zoom_range = 0.20,
fill_mode = "nearest", shear_range = 0.20, horizontal_flip = True,
width_shift_range = 0.1, height_shift_range = 0.1)
history = model.fit_generator(dataAugmentaion.flow(x_train, y_train, batch_size=batch),
                              validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // batch,
                              epochs=int(epoch), verbose=1, shuffle=True, callbacks=cbs)'''
start_time = time.process_time()
def run():
    # call logger
    allocate_gpu_memory()

    # data load
    from train_image_read import clean_train, noisy_train
    x_train, x_test, y_train, y_test = train_test_split(noisy_train, clean_train, test_size=0.25, random_state=42)
    clean_train = 0
    noisy_train = 0
    print('x y train shape', x_train.shape, y_train.shape)
    print('x y test shape', x_test.shape, y_test.shape)

    # define model
    #model = bm_model.unet()
    model = bm_model.AttentDenseUNet()
    print(model.summary())

    # compile
    learning_rate = 1e-3 # 0.0001 #1e-4 # 0.0005 # 0.00001
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9,beta_2=0.999)
    model.compile(optimizer=optimizer, loss= "mse", metrics= ['acc'])

    batch = 2
    epoch = '1000'

    # callbacks
    model_name = 'AttentDenseUNet'
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    output = pathlib.Path(f'./{model_name}/{now}_{epoch}')
    output.mkdir(exist_ok=True, parents=True)

    tensorboard = TensorBoard(log_dir=f'{output}/logs')

    cp = ModelCheckpoint(filepath=f'{output}/AttentDenseUNet_linear_1024.h5',
                         monitor='val_loss',
                         save_best_only=True,
                         save_weights_only=False,
                         verbose= 1,
                         mode='auto')

    logger = CSVLogger(f'{output}/history.csv')
    # Step Decay: Step decay schedule drops the learning rate by a factor every few epochs.
    def step_decay(epoch):
        return learning_rate * 0.95 ** (epoch // 30)

    # decay_rate = learning_rate / int(epoch) #default = 1e-7
    # def exp_decay(epoch):
    #     return learning_rate * np.exp(-decay_rate * epoch)

    lr_decay = LearningRateScheduler(step_decay, verbose=1)
    cbs = [cp, logger, lr_decay, tensorboard]

    # START training
    history = model.fit(x_train, y_train, batch_size = batch,
        epochs=int(epoch),
        verbose=1,
        shuffle = True,
        callbacks=cbs,
        validation_data = (x_test, y_test))

    end_time = time.process_time() - start_time
    hours, rem = divmod(end_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time elapsed for training: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    _, acc = model.evaluate(x_test, y_test)
    print('Accuracy = ', (acc * 100.0), '%')

    #print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(x_test), np.array(y_test))[1] * 100))
    # plot loss during training
    pyplot.title('Loss / Mean Squared Error')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

if __name__ == "__main__":
    run()

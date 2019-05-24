import os       
import cv2
import numpy as np
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import *
import matplotlib.pyplot as plt
from keras.regularizers import *
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from time import time
from keras.preprocessing.image import ImageDataGenerator
# from gray_model import *
from GAN_models_line import *
from losses import *


##This is to prevent out of memory error on cuda 10
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


#seed
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

#dimensions
x_shape = 512
y_shape = 512

batch = 150

#directories
val_set = "../data/validation_set_line"
data_out = "../data/data_out_test/"

def GetDataset(dataset, rgb ,gray):
    for root, dirs, files in os.walk(dataset):
        for image in files:
            temp = cv2.imread(root+'/'+image)
            temp = cv2.resize(temp, (x_shape, y_shape))
            rgb.append(temp)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
            gray.append(temp)


#setup models
gen = generator_model(x_shape,y_shape)
disc = discriminator_model(x_shape,y_shape)
advr = advr_model(gen,disc)

advr.load_weights("../data/data_out/updated.h5")

#setup dataset
val_rgb = []
val_gray = []


print("Loading dataset")
#get dataset
GetDataset(val_set, val_rgb, val_gray)

#convert to numpy array
val_rgb = np.array(val_rgb)
val_gray = np.array(val_gray)

#normalize rgb to match the output range of -1 to 1
val_gray = (val_gray-127.5)/127.5
print("dataset loaded")


gen_image_val = (gen.predict(val_gray, batch_size=8)*127.5)+127.5
for j in range(len(gen_image_val)):
    cv2.imwrite(data_out +str(j)+'.jpg', gen_image_val[j])
    



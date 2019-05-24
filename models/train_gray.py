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
from GAN_models import *
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
color_set = "../data/color_set/"
val_set = "../data/validation_set/"
data_out = "../data/data_out/"

def GetDataset(dataset, rgb ,gray):
    for root, dirs, files in os.walk(dataset):
        for image in files:
            temp = cv2.imread(root+image)
            temp = cv2.resize(temp, (x_shape, y_shape))
            rgb.append(temp)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
            gray.append(temp)


#setup models
gen = generator_model(x_shape,y_shape)
# gen.compile(loss=[custom_loss_2], loss_weights = [100] , optimizer=Adam(lr=2E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])
# gen.summary()

disc = discriminator_model(x_shape,y_shape)
disc.trainable = False
advr = advr_model(gen,disc)
advr.compile(loss=['binary_crossentropy',custom_loss_2], loss_weights = [5,100] , optimizer=Adam(lr=2E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])
advr.summary()

disc.trainable = True
disc.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=2E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])
disc.summary()

advr.load_weights("../data/data_out/updated.h5")

#setup dataset
rgb = []
gray = []
val_rgb = []
val_gray = []

# rgb = np.zeros((samples, x_shape, y_shape, 3))
# gray = np.zeros((samples, x_shape, y_shape, 1))
# val_rgb = np.zeros((val_samples, x_shape, y_shape, 3))
# val_gray = np.zeros((val_samples, x_shape, y_shape, 1))

print("Loading dataset")
#get dataset
GetDataset(color_set, rgb, gray)
GetDataset(val_set, val_rgb, val_gray)

#convert to numpy array
rgb = np.array(rgb)
gray = np.array(gray)
val_rgb = np.array(val_rgb)
val_gray = np.array(val_gray)

#normalize rgb to match the output range of -1 to 1
rgb = (rgb-127.5)/127.5
gray = (gray-127.5)/127.5
val_gray = (val_gray-127.5)/127.5
print("dataset loaded")

#training
e = 0
while 1:
    if e%200 == 0:
        rand_indexes = np.random.randint(0, len(rgb), size=batch)
        sel_gray = gray[rand_indexes]
        sel_rgb = rgb[rand_indexes]

        for i in range(len(sel_rgb)):
            if not os.path.exists(data_out+'batch/'):
                os.mkdir(data_out+'batch/')
            if not os.path.exists(data_out+'batch/'+str(e)+'/'):
                os.mkdir(data_out+'batch/'+ str(e)+'/')
            cv2.imwrite(data_out+'batch/'+ str(e)+'/'+str(i)+'.jpg', (sel_rgb[i]*127.5)+127.5) 

    print('Epoch ', e)
    # gen.fit(sel_gray,sel_rgb, epochs=1, batch_size=1)

    gen_image = gen.predict(sel_gray, batch_size=16)
    inputs = np.concatenate([sel_gray, sel_gray])
    outputs = np.concatenate([sel_rgb[:,:,:,:-1], gen_image])
    y = np.concatenate([np.ones((batch,1)), np.zeros((batch,1))])
    disc.fit([inputs, outputs], y, epochs=1, batch_size=4)
    disc.trainable = False
    advr.fit(sel_gray, [np.ones((batch,1)),sel_rgb[:,:,:,:-1]], epochs=1, batch_size=1)
    disc.trainable = True

    if e%5 == 0:
        advr.save_weights(data_out+'updated.h5') 
    gen_image_val_2 = (gen.predict(val_gray, batch_size=8)*127.5)+127.5
    # ave = (gen_image_val[:,:,:,0] +  gen_image_val[:,:,:,1] +  gen_image_val[:,:,:,2])/3
    # for i in range(3):
    #     gen_image_val[:,:,:,i] = gen_image_val[:,:,:,i]*((val_gray[:,:,:,0]*127.5)+127.5)/ave
    gen_image_val = np.zeros((len(gen_image_val_2), gen_image_val_2.shape[1], gen_image_val_2.shape[2], 3))
    gen_image_val[:,:,:,:-1] = gen_image_val_2
    gen_image_val[:,:,:,2] = ((val_gray[:,:,:,0]*127.5)+127.5)*3  - gen_image_val[:,:,:,0] - gen_image_val[:,:,:,1]

    # print(val_gray[:,:,:,0]*3)
    # print(val_rgb[0])
    np.clip(gen_image_val,0,255)
    for j in range(len(gen_image_val)):
        # for ii in range(gen_image_val.shape[1]):
        #     for jj in range(gen_image_val.shape[2]):
        #         rg = gen_image_val[j][ii][jj][0] + gen_image_val[j][ii][jj][1]
        #         gr = val_gray[j][ii][jj][0]*3
        #         if  rg < gr:
        #             gen_image_val[j][ii][jj][2] = gr - rg
        #         else:
        #             gen_image_val[j][ii][jj][2] = 0
        
        if not os.path.exists(data_out+str(j)+'/'):
            os.mkdir(data_out+str(j)+'/')
        cv2.imwrite(data_out+str(j)+'/'+str(e)+'.jpg', gen_image_val[j])
    # 
    e += 1


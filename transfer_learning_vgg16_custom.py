import numpy as np
import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten, merge, Input
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split



path=os.getcwd
data_path=path + "/data"
data_dir_list=os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
    images=os.listdir(data_path+"/"+dataset)
    for img in images:
        img=image.load_img(images+"/"+img,target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        print("input image shape",x.shape)
        img_data_list.append(x)
        
        
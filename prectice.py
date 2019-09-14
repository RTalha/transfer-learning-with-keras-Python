import numpy as np
from models.vgg16 import VGG16
from keras.preprocessing import image
from models.imagenet_utils import preprocess_input ,decode_predictions


model=VGG16(include_top=True,weights='imagenet')

img=image.load_img("elephent.jpeg",target_size=(224,224))

x=image.img_to_array(img)

print(type(x))
x = np.expand_dims(x,axis=0)
x=preprocess_input(x)

predict=model.predict(x)

print("prediction",decode_predictions(predict))

model.summary()


model.layers[-1].get_config()

//////////////////////////////////////////////


model=VGG16(weights='imagenet',include_top=False)
model.summary()


model.layers[-1].get_config()

predict=model.predict(x)

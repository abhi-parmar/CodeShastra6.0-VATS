import keras
import tensorflow as tf
import cv2
import numpy as np


def get_classlabel(class_code):
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}
   
    return labels[class_code]

model =tf.keras.models.load_model('/home/trunesh/Downloads/model.hdf5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('/home/trunesh/Downloads/64.jpg')
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

classes = model.predict_classes(img)
label=get_classlabel(classes[0])
print("\nthis image is of a "+label+"\n")

imgs=[]

import os
for r,d,f in os.walk('/home/trunesh/all_images'):
for fi in f:
print(os.path.join(r,fi))
tpath=os.path.join(r,fi)
imgs.append(tpath)

for path in imgs:
img = cv2.imread(path)
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

classes = model.predict_classes(img)
label=get_classlabel(classes[0])
print("\nthis image is of a "+label+"\n"
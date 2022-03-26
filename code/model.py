import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

batchSize= 32
testSteps=(24,24)

train_batch= generator('data/train',shuffle=True, batch_size=batchSize,target_size=testSteps)
valid_batch= generator('data/test',shuffle=True, batch_size=batchSize,target_size=testSteps)
print(train_batch.target_size)
print(valid_batch.target_size)


# you should reshape your labels as 2d-tensor
# the first dimension will be the batch dimension and the second the scalar label)

#train_batch = np.asarray(train_batch).astype('float32').reshape((-1,1))
#valid_batch = np.asarray(valid_batch).astype('float32').reshape((-1,1))


stepsPerEpoch= len(train_batch.classes)//batchSize
validSteps = len(valid_batch.classes)//batchSize
print(stepsPerEpoch,validSteps)


# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

#64 convolution filters used each of size 3x3
#choose the best features via pooling
    
#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(128, activation='relu'),
#one more dropout for convergence' sake :) 
    Dropout(0.5),
#output a softmax to squash the matrix into output probabilities
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=stepsPerEpoch ,validation_steps=validSteps)

#model.save('models/cnnCat3.h5', overwrite=True)
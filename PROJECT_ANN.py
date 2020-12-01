import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pickle
import random

with open("./traffic-signs-data/train.p",mode='rb')as training_data:
    train = pickle.load(training_data)

with open("./traffic-signs-data/valid.p",mode='rb')as validation_data:
    valid = pickle.load(validation_data)

with open("./traffic-signs-data/test.p",mode='rb')as testing_data:
    test = pickle.load(testing_data)


x_train,y_train =train['features'],train['labels']

x_train.shape
y_train.shape

x_validation,y_validation =valid['features'],valid['labels']

x_validation.shape
y_validation.shape

x_test,y_test =test['features'],test['labels']

x_test.shape
y_test.shape

#Task 03 Perform Images Visualization

i=np.random.randint(1,len(x_train))
plt.imshow(x_train[i])
y_train[i]

#Lets view more images in a grid format
#Define the dimensions of the plot grid

w_grid=5
l_grid=5

fig,axes=plt.subplots(l_grid,w_grid,figsize=(10,10))

axes=axes.ravel()
n_training =len(x_train)

for i in np.arange(0,w_grid*l_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(x_train[index])
    axes[i].set_title(y_train[index],fontsize=15)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)

from sklearn.utils import shuffle
x_train, y_train =shuffle(x_train,y_train)
x_train_gray = np.sum(x_train/3,axis=3,keepdims=True)
x_train_gray.shape
x_train_gray_norms =(x_train_gray-128)/128
x_train_gray_norms


x_validation, y_validation =shuffle(x_validation,y_validation)
x_validation_gray = np.sum(x_validation/3,axis=3,keepdims=True)
x_validation_gray.shape
x_validation_gray_norms =(x_validation_gray-128)/128
x_validation_gray_norms

x_test, y_test =shuffle(x_test,y_test)
x_test_gray = np.sum(x_test/3,axis=3,keepdims=True)
x_test_gray.shape
x_test_gray_norms =(x_test_gray-128)/128
x_test_gray_norms


i=random.randint(1,len(x_train_gray))
plt.imshow(x_train_gray[i].squeeze(),cmap='gray')
plt.figure()
plt.imshow(x_train[i])
plt.figure()
plt.imshow(x_train_gray_norms[i].squeeze(),cmap='gray')


#ANN CODE

from tensorflow.keras import datasets,layers,models
CNN=models.Sequential()

CNN.add(layers.Conv2D(6,(5,5), activation ='relu',input_shape =(32,32,1)))
CNN.add(layers.AveragePooling2D())
CNN.add(layers.Dropout(0.2))
CNN.add(layers.Conv2D(6,(5,5), activation ='relu'))
CNN.add(layers.AveragePooling2D())


CNN.add(layers.Flatten())

CNN.add(layers.Dense(120,activation = 'relu'))
CNN.add(layers.Dense(84,activation='relu'))
CNN.add(layers.Dense(43,activation='softmax'))
print(CNN.summary)

CNN.compile(optimizer ='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history=CNN.fit(x_train_gray_norms,
                y_train,
                batch_size =500,
                epochs=5,
                verbose=1,
                validation_data=(x_validation_gray_norms, y_validation))


score=CNN.evaluate(x_test_gray_norms,y_test)
print('test accuracy:{}',format(score[1]))
history.history.keys()
accuracy = history.history['accuracy']
loss=history.history['loss']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

epochs=range(len(accuracy))
plt.plot(epochs,loss,'ro',label='training loss')
plt.plot(epochs,val_loss,'r',label='validation loss')
plt.title('taining and validation loss')

epochs=range(len(accuracy))
plt.plot(epochs,accuracy,'ro',label='training accuracy')
plt.plot(epochs,val_accuracy,'r',label='validation accuracy')
plt.title('taining and validation accuracy')


predicted_classes =CNN.predict_classes(x_test_gray_norms)
y_true=y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true ,predicted_classes)
plt.figure(figsize=(25,25))
sns.heatmap(cm,annot=True)
L=5
W=5

fig,axes =plt.subplots(L,W,figsize=(12,12))
axes = axes.ravel()

for i in np.arange(0,L*W):
    axes[i].imshow(x_test[i])
    axes[i].set_title('Prediction={}\n True ={}'.format(predicted_classes[i],y_true[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=1)



plt.show()






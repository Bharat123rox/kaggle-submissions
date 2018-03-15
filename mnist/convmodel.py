import pandas as pd
import os
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential, save_model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.datasets import mnist
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.models import save_model
from keras import backend as K
rows,cols=28,28
batch=128
classes=10
epochs=50
labeled_images = pd.read_csv('train.csv')
images = labeled_images.drop('label',axis=1)
labels = labeled_images['label']
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, test_size=0.2, random_state=42)
train_images = train_images.as_matrix()
test_images = test_images.as_matrix()
if K.image_data_format() == 'channels_last':
	train_images = train_images.reshape(train_images.shape[0],rows,cols,1).astype('float32')
	test_images = test_images.reshape(test_images.shape[0],rows,cols,1).astype('float32')
	dims = (rows,cols,1)
else:
	train_images = train_images.reshape(1,train_images.shape[0],rows,cols).astype('float32')
	test_images = test_images.reshape(1,test_images.shape[0],rows,cols).astype('float32')
	dims = (1,rows,cols)
train_images /= 255
test_images /= 255
train_labels = to_categorical(train_labels, classes)
test_labels = to_categorical(test_labels, classes)
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=dims))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adadelta(),metrics=['accuracy'])
model.fit(train_images,train_labels,batch_size=batch,epochs=epochs,verbose=2,validation_split=0.1)
if not os.path.exists(os.path.join(os.getcwd(),'convmodel.h5')):
	model_path = os.path.join(os.getcwd(),'convmodel.h5')
	model.save(model_path)
score = model.evaluate(test_images, test_labels, verbose=1)
print('Validation test loss:', score[0])
print('Validation test accuracy:', score[1])
test_data = pd.read_csv('test.csv').astype('float32')
test_data /= 255
results = model.predict(test_data)
df = pd.DataFrame(results)
df.index.values = 'ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
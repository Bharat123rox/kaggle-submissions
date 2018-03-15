import pandas as pd
import numpy as np
import os
import keras
from sklearn.model_selection import train_test_split
from keras.models import load_model, Sequential
model = Sequential()
model_path = os.path.join(os.getcwd(),'convmodel.h5')
model = load_model(model_path)
test_data = pd.read_csv('test.csv').astype('float32')
test_data /= 255
test_data = test_data.as_matrix()
test_data = test_data.reshape(test_data.shape[0],28,28,1)
results = model.predict(test_data)
results = np.argmax(results,axis=1)
df = pd.DataFrame(results)
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
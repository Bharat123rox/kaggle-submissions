import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn import svm

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[:,1:].astype('float32')
labels = labeled_images.iloc[:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, test_size=0.2, random_state=42)
train_images /= 255
test_images /= 255
# param_grid = {
# 	'C':[1,10,100,1000],
# 	'kernel':('linear','rbf'),
# 	'gamma':[0.001,0.01,0.1] }
clf = svm.SVC(C=10,kernel='rbf',gamma=0.01)
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images,test_labels))
test_data = pd.read_csv('test.csv').astype('float32')
test_data /= 255
results = clf.predict(test_data)
df = pd.DataFrame(results)
df.index.values = 'ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
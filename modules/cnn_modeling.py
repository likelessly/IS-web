import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
import seaborn as sns
import numpy as np

np.random.seed(0)

from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


num_classes=10
f,ax=plt.subplots(1,num_classes,figsize=(20,20))
for i in range(0,num_classes):
  sample=x_train[y_train==i][0]
  ax[i].imshow(sample,cmap='gray')
  ax[i].set_title("label:{}".format(i),fontsize=16)

for i in range(10):
  print(y_train[i])
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=10)
# Also convert y_test to categorical format
y_test = to_categorical(y_test, num_classes=10)

#Normalize data
x_train=x_train/255.0
x_test=x_test/255.0
#reshape data
x_train=x_train.reshape(x_train.shape[0],-1)  #flattering the data from 2d/3d to 1d
x_test=x_test.reshape(x_test.shape[0],-1)
print(x_train.shape)
model=Sequential() #This creates a linear stack of layers
model.add(Dense(units=128,input_shape=(784,),activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
batch_size=512
epochs=10
model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs)
test_loss,test_acc=model.evaluate(x_test,y_test)
print("test loss:{},test accuracy:{}".format(test_loss,test_acc))
plt.imshow(x_test[0], cmap='gray')
yp = model.predict(x_test)
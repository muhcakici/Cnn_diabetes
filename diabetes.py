import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
dataset=np.loadtxt('diabetes1.csv',delimiter=',')

x=dataset[:,:8]
y=dataset[:,8]

from sklearn.model_selection import train_test_split
training_set_x,test_set_x,training_set_y,test_set_y=train_test_split(x,y,test_size=0.2)

from keras.models import Sequential
from keras.layers import Dense
model =Sequential()
model.add(Dense(50,input_dim=8,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentrapy',optimizer='adam',metrics=['accuracy'])
model.fit(training_set_x,training_set_y,epochs=10,batch_size=8)

test_result=model.evaluate(test_set_x,test_set_y)
print('test loss,test acc:',test_result)



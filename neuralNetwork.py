
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense


dataset = pd.read_csv("diabetes.csv")

dataset


X = dataset.iloc[:,:8].values
y = dataset.iloc[:,8:9].values


sc = StandardScaler()
X = sc.fit_transform(X)
# y = sc.fit_transform(y)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y = ohe.fit_transform(y).toarray()

y


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

len(y_test)




model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=64 )


y_pred = model.predict(X_test)



pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))



test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))


a = accuracy_score(pred,test)
print('Accuracy:', a*100)
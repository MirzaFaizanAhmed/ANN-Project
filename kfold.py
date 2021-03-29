import numpy as np    
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler



seed = 7
np.random.seed(seed)

dataset = pd.read_csv("diabetes.csv")

X = dataset.iloc[:,:8].values
y = dataset.iloc[:,8:9].values


sc = StandardScaler()
X = sc.fit_transform(X)




kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []


for train, test in kfold.split(X,y):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X[train], y[train], epochs=150, batch_size=10, verbose=0)
    scores = model.evaluate(X[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
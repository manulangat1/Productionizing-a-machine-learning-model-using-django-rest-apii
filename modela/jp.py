from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

iris = load_iris()
x = iris.data
y = iris.target
# print(x)
x_df = pd.DataFrame(data=x,columns=iris.feature_names)
# print(x_df.describe())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=77)
model =  RandomForestClassifier(random_state=1,n_estimators=10,n_jobs=2)
model.fit(x_train,y_train)
# predict = model.predict(x_test)
# print(model.score(x_test,y_test))
a = cross_val_score(model,x,y,cv=5,scoring='accuracy')
print(a.mean())

from joblib import dump, load
dump(model, 'IRISRandomForestClassifier.joblib')
loaded_classifier = load('IRISRandomForestClassifier.joblib')
b = loaded_classifier.score(x_test,y_test)
print(b)
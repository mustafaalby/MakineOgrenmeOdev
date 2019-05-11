#%%
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
#%%
y=pd.read_csv("colorectal_cancer_labels.csv")
x=pd.read_csv("colorectal_cancer.csv")
#%%
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=1)
#%%
trained_Model=XGBClassifier()
trained_Model.fit(xTrain,yTrain)
#%%
pred=trained_Model.predict(xTest)
prediction = [round(value) for value in pred]
#%%
acc = accuracy_score(yTest,prediction)
print("Accuracy : %.3f" % (acc * 100))



#%%

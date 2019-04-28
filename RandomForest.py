#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#%%
y=pd.read_csv("colorectal_cancer_labels.csv")
x=pd.read_csv("colorectal_cancer.csv")

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.10,random_state=1)
#%%
RF=RandomForestClassifier(n_estimators=500,random_state=61)
RF.fit(xTrain,yTrain)
print("Score", RF.score(xTest,yTest))

#%%

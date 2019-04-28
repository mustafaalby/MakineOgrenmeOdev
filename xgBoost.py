#%%
import numpy as np
import pandas as pd
import xgboost as xgb
#%%
y=pd.read_csv("colorectal_cancer_labels.csv")
x=pd.read_csv("colorectal_cancer.csv")
concat=pd.concat([x,y],axis=1)
#%%
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=1)
#Train=pd.concat([xTrain,yTrain],axis=1)
#Test=pd.concat([xTest,yTest],axis=1)
dTrain = xgb.DMatrix(xTrain, label=yTrain)
dTest = xgb.DMatrix(xTest, label=yTest)
#%%
param = {'max_depth':3, 'eta':0.3,
         'silent':1,
         'objective':'multi:softprob','num_class':4 }
num_round = 2
bst=xgb.train(param,dTrain,num_round)
#%%
predictions=bst.predict(dTest)



#%%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix,accuracy_score
df=pd.read_csv('Social_Network_Ads.csv')
x=df.iloc[:,:-1] 
y=df.iloc[:,-1] 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=10,random_state=0)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
#prediction
model=SVC(kernel='linear', random_state=0)
model.fit(x_train,y_train) 
pred=model.predict(x_test)
cm=confusion_matrix(y_test,pred)
print(cm)
ac=accuracy_score(y_test,pred)
print(ac)
print(model.predict(sc.transform([[33,49000]]))[0])
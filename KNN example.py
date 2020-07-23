"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/krishnaik06/K-Nearest-Neighour/master/Classified%20Data",index_col=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

#TO CHOOSE ACCURATE K-VALUE:
    
accuracy_rate = []
for i in range(1,40):    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    accuracy_rate.append(score.mean())
    
error_rate = []
for i in range(1,40):    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    error_rate.append(1-score.mean())
    
plt.figure(figsize=(10,6))
#plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy rate')    
#.show()
    
#THUS WE FOUND AT K=23, THE ACCRACY RATE INCREASES, SO WE CAN APPLY K=2 IN THE KNN ALGORITHM AND AGAIN FIND AN ACCURACY SCORE

knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
"""

#DIABETES DATASET


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
dataset=pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')
#to replace zeros with mean of that column
zero_not_accepted=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for column in zero_not_accepted:
    dataset[column]=dataset[column].replace(0,np.NaN)
    mean=int(dataset[column].mean(skipna=True))
    dataset[column]=dataset[column].replace(np.NaN,mean)
X=dataset.iloc[:,0:8]
y=dataset.iloc[:,8]
sc=StandardScaler()
X=sc.fit_transform(X)
X=sc.transform(X)
accuracy_rate = []
for i in range(1,40):    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X,y,cv=10)
    accuracy_rate.append(score.mean())
"""
plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Accuracy rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy rate')   
plt.show()
"""
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)
classifier=KNeighborsClassifier(n_neighbors=34,p=2,metric='euclidean')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('For k=34:')
print(cm)
print(accuracy_score(y_test,y_pred))


 
    
    
    
    
    
    
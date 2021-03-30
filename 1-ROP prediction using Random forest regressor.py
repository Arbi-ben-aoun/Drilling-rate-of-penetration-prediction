#Import data analysis python libraries
Import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

# import drilling data 
welldf=pd.read_csv('Well_58-32.csv')
welldf.head()
welldf.describe().transpose()
welldf.describe().transpose().to_csv('decription.csv', sep=',')
welldf.columns
df=welldf[['ROP(1 m)','Depth(m)','weight on bit (kg)','Rotary Speed (rpm)','Pump Press (KPa)','Temp In(degC)',   'Flow In(liters/min)','Flow Out %']]
df.to_csv('well.csv', index=False) 

# Exploratory data analysis
#cheking for missing values 
df.isnull().sum()
plt.figure(figsize=(15,5))
sns.heatmap(data=df,cmap='magma')
#cheking for correlation between features and outliers
sns.pairplot(df)
plt.figure(figsize=(15,5))
sns.heatmap(df.corr())


#which are the most correlated features with our target variable ROP
df.corr()['ROP(1 m)'].sort_values
#Outlier removal 
plt.figure(figsize=(20,8))
sns.boxplot(df['ROP(1 m)'])
df[df['ROP(1 m)'] > 800]
df=df.drop(index=259,axis=0)
sns.boxplot(df['Rotary Speed (rpm)'])
df[df['Rotary Speed (rpm)'] > 200]
df=df.drop(index=[245,246,247,248,249,252],axis=0)
sns.boxplot(df['Flow In(liters/min)'])
df[df['Flow In(liters/min)'] > 7000]
df=df.drop(index=[2462,4152],axis=0)
df[df['Flow In(liters/min)'] > 5000]
sns.boxplot(df['Flow Out %'])
df[df['Flow Out %'] < 10 ]
df=df.drop(index=[1720,1721,2998,3260,3261,3262,3411,3572,4561],axis=0)
#checking the data after outlier removal
sns.pairplot(df)
df.describe()
#saving backup after outlier removal
df.to_csv('well_without_outlier.csv',index=False)

#import Scikit-learn machine learning libraries
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

#create our dependent and independent variables
y=df['ROP(1 m)']
X=df.drop(['ROP(1 m)'],axis=1)
#check the dimensions
print('the dataframe dimension is :',df.shape)
print('Y dimension is :',y.shape)
print('X  dimension is :',X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
#random try of random forest on the training set
rfran.fit(X_train,y_train)
predict=rfran.predict(X_test)
mse=mean_squared_error(y_test,predict)
print('the test set mean squared error is {}'.format(mse))
importance=pd.Series(data=rfran.feature_importances_,index=X.columns)
sorted_importance=importance.sort_values(ascending=True)

#feature importance based on tree models
plt.figure(figsize=(10,5))
sorted_importance.plot(kind='barh',color='red')
plt.title('Feature importance') 
plt.xlabel('importance score') 
plt.ylabel('Feature name')
cumsortimp=np.cumsum(sorted_importance.sort_values(ascending=False))
plt.figure(figsize=(13,5))
plt.plot(cumsortimp.index,cumsortimp[:],color='red')
plt.hlines(y=0.95,xmin=cumsortimp.index[0],xmax=cumsortimp.index[-1],linestyles='dashed')
plt.title('Cumulative Feature importance') 
plt.xlabel('Cumulative importance score') 
plt.ylabel('Feature name')

#we can remove flowout feature  
important_features=cumsortimp.drop(labels='Flow Out %')
important_X_train = X_train[important_features.index]
important_X_test=X_test[important_features.index]
print('when considering only important features the training data set dimension is {}'.format(important_X_train.shape))
print('when considering only important features the test dataset dimension is {}'.format(important_X_test.shape))

# Scaling training data
#we apply min-max normalization for the features
Xscale=MinMaxScaler().fit(important_X_train)
scaled_Xtrain=Xscale.transform(important_X_train)
scaled_Xtest=Xscale.transform(important_X_test)
# Random forest regression
rf=RandomForestRegressor()
#hyperparameter tuning using gridsearchCV
rfparam={'n_estimators':[300,400],'max_depth':[5,10,None]}
gridrf=GridSearchCV(estimator=rf,param_grid=rfparam,cv=5,verbose=3,scoring='neg_mean_absolute_error')
gridrf.fit(scaled_Xtrain,y_train)
gridrf.best_estimator_
best_rf=gridrf.best_estimator_
predictedrf=best_rf.predict(scaled_Xtest)
print('root mean squared error is {}'.format(np.sqrt(mean_squared_error(y_test,predictedrf))))
print('Mean absolute error is {}'.format(mean_absolute_error(y_test,predictedrf)))
print('R2 score is {}'.format(r2_score(y_test,predictedrf)))
#plot predicted vs true R.O.P
plt.figure(figsize=(13,6))
plt.title('Mean absolute error is {} and R2 score is {}'.format(mean_absolute_error(y_test,predictedrf),r2_score(y_test,predictedrf)))
plt.plot(np.arange(0,300,1),np.arange(0,300,1),color='green', linestyle='dashed',label='Predicted R.O.P = True R.O.P')
sns.scatterplot(x=y_test,y=predictedrf)
plt.xlabel('True R.O.P')
plt.ylabel('Predicted')

#ROP vs Depth with random forest predictions
fig,ax=plt.subplots(figsize=(15,25))
ax.plot(y,X['Depth(m)'].values.reshape(-1,1),'r',label='ROP')
ax.scatter(predictedrf,X_test['Depth(m)'].values.reshape(-1,1),label='predicted ROP')
ay=plt.gca()
ay.set_ylim(ay.get_ylim()[::-1])
plt.ylabel('Depth (ft)')
plt.xlabel('Rate of penetration (m/hr)')
plt.title('Rate of penetration prediction')
plt.legend(loc="best")
depth=[depth for depth in range(200,2400,200) ]
for i in range(len(depth)):
    plt.axhline(depth[i],color='black' )
plt.savefig('Random forest resgression.png') 

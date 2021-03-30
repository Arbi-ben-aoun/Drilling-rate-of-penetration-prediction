#Import data analysis and deep learning Keras libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,load_model,save_model
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

#import preprocessed well data 
df=pd.read_csv('well_without_outlier.csv')
df=df.drop('Flow Out %',axis=1)
#Statistical description of the selected features for the target variable ROP
df.describe()
X=df.drop('ROP(1 m)',axis=1)
y=df['ROP(1 m)']
print('the Shape of x is :',X.shape)
print('the Shape of y is :',y.shape)

#Split of target variable and predictors 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#convert the normalized features into numpy array 
scaler=MinMaxScaler().fit(X_train)
scaled_X_train=np.array(scaler.transform(X_train))
scaled_X_test=np.array(scaler.transform(X_test))
#convert the target variable into numpy array 
ny_train=np.array(y_train)
ny_test=np.array(y_test)

#intialize a sequential model to be tuned later
def ANN_model(nl=0,nn=6):
    model = Sequential()
    model.add(Dense(6,input_dim=6,activation="relu"))
    for i in range(nl):
        model.add(Dense(nn, activation='relu'))  
   
 model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam',metrics=['mse'] )
    return model         
ANN_model().summary()

#create a KerasRegressor based on this ANN_model()
Ann_reg=KerasRegressor(build_fn=ANN_model,epochs=200, batch_size=16)
#choosing the number of layers and number of neurons as hyperparameters 
param={'nl':[0,1,2],'nn':[2,3,6,12,24]}
grid_cv = GridSearchCV(Ann_reg, param_grid=param, cv=5,verbose=2,scoring='neg_mean_absolute_error')
grid_cv.fit(scaled_X_train, ny_train,validation_split=0.3 ,callbacks=[EarlyStopping(patience=10)])
#Note that GridSearchCV uses K-fold cross-validation, so it does not use validation split, These are just used for early stopping
#the best hyperparameter values for ANN_model
grid_cv.best_params_
best_ann=grid_cv.best_estimator_
#Now try to fit the model by using callbacks , monitor the validation loss and save only the best model
best_ann.fit(scaled_X_train, ny_train,validation_split=0.2,verbose=2,callbacks=[EarlyStopping(patience=10),ModelCheckpoint('best_model.hdf5',save_best_only=True)])
#loading the model from checkpoint
best_ann=load_model('best_model.hdf5')

#the best ANN architechture
best_ann.summary()
predictions=best_ann.predict(scaled_X_test)
#evaluate the model metrics
print('mean absolut error', mean_absolute_error(ny_test,predictions)) 
print('R2 score',r2_score(ny_test,predictions)) 
print('root mean squared error',np.sqrt(mean_squared_error(ny_test,predictions)))

#Plot predicted vs true R.O.P
plt.figure(figsize=(13,8))
plt.title('Mean absolute error is {} and R2 score is {}'.format(mean_absolute_error(y_test,predictions),r2_score(y_test,predictions)))
plt.plot(np.arange(0,300,1),np.arange(0,300,1),color='green', linestyle='dashed',label='Predicted R.O.P = True R.O.P')
sns.scatterplot(x=ny_test,y=predictions[:,0])
plt.xlabel('True R.O.P')
plt.ylabel('Predicted')
predictions[:,0]


# Examining the best model training history
final_model=Sequential()
final_model.add(Dense(6,input_dim=6,activation="relu"))
final_model.add(Dense(12,activation="relu"))
final_model.add(Dense(12,activation="relu"))
final_model.add(BatchNormalization())
final_model.add(Dense(1))
final_model.compile(optimizer="adam", loss="mse", metrics = ['mae'])
final_model.fit(scaled_X_train, ny_train, batch_size=16 ,epochs=200,validation_split=0.2,verbose=2)

#by plotting the learning curves of the neural network we can see that 100 epochs are enough train it 
plt.figure(figsize=(15,8))
# Use the history metrics
plt.plot(final_model.history.history['loss'])
plt.plot(final_model.history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Mean squared error')
plt.legend(['Train', 'validation'])
plt.show()

#Plot R.O.P vs depth with Neural network predictions
fig,ax=plt.subplots(figsize=(15,30))
ax.plot(y,X['Depth(m)'].values.reshape(-1,1),'r',label='ROP')
ax.scatter(predictions,X_test['Depth(m)'].values.reshape(-1,1),label='predicted ROP')
ay=plt.gca()
ay.set_ylim(ay.get_ylim()[::-1])
plt.ylabel('Depth (ft)')
plt.xlabel('Rate of penetration (m/hr)')
plt.title('Rate of penetration prediction');
plt.legend(loc="best")
depth=[depth for depth in range(200,2400,200) ]
for i in range(len(depth)):
    plt.axhline(depth[i],color='black' )

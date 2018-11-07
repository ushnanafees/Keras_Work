import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
from keras import Sequential

#------------------------------------- Load Dataset ------------------------------------
DS = np.loadtxt("E:/SSUET_WORK/PIMA Diabetes/diabetes.csv", delimiter = ",")

#------------------------------------ SPlit Dataset ------------------------------------
X = DS[:, 0:8]           # Features
Y = DS[:,8]              # Labels
print(X.shape, Y.shape)


#------------------------------------ Build Model --------------------------------------
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(12, input_dim=8, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss ='binary_crossentropy' , 
              optimizer = 'adam',
              metrics = ['accuracy'])
    return model


#--------------------------------- K-Fold Validation ----------------------------------
k = 4
num_validation_samples = len(DS) // k
np.random.shuffle(DS)
validation_scores = []
for fold in range(k):
    validation_data = DS[num_validation_samples * fold:
    num_validation_samples * (fold + 1)]
    training_data = np.concatenate ((DS[:num_validation_samples * fold] , DS[num_validation_samples * (fold + 1):]) , axis = 0)
    
    X = training_data[: , 0:8]
    Y = training_data[: , 8]
    model = build_model()
    model.fit(X , Y , epochs = 50 , batch_size = 15)   
    x_val = validation_data[: ,0 :8]
    y_val = validation_data[: ,8]
    val = model.evaluate(x_val,y_val)
    print(val)


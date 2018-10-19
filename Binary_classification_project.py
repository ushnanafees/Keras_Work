import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed
seed = 7
np.random.seed(seed)

# Load dataset
dataframe = pd.read_csv("E:\BSCS 7\Practice_work\Deep Learning\Deep_Learning_Project_One\sonar.csv", header = None)
dataset = dataframe.values
#print(dataset)
# split into input (X) and Output (Y)
X = dataset[:,0:60].astype(float)
#print(X)
Y = dataset[:,60]
#print(Y)

# evaluate model with standardized dataset
def create_baseline():
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
    print("Results: %.2f%%(%.2f%%)" %(results.mean()*100,results.std()*100))
    return create_baseline()

def create_smaller():
    np.random.seed(seed)
    estimators = []
    estimators.append(('standarize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
    print("Results: %.2f%%(%.2f%%)" %(results.mean()*100,results.std()*100))
    return create_smaller()








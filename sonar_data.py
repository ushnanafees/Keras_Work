import numpy
import pandas as pd
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load dataset
dataframe = pd.read_csv("E:\BSCS_7\Practice_work\Deep_Learning\Deep_Learning_Project_One\sonar.csv", header = None)# header=None is waja s lya hy qk dataset m column names nae hyn..
dataset = dataframe.values #only the values in dataframe will returned, axes labels (x,y-axis) will be removed..
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float) # sari rows, 0 s 59 tk k columns., sb rows columns  float m convert hojaygy
Y = dataset[:,60] # sari rows only 60th column ki. Output variable last column string hy (M,R) m usy integer value (0 and 1) m convert karygy



from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(Y)
x = list(le.classes_)
x = le.transform(Y)
print(x)
 

### baseline model
#def create_baseline():
#   model = keras.Sequential()    # linear stack, hr layer k bd ek he layer aygi   
#   model.add(layers.Dense(16, activation='relu'))
#   model.add(layers.Dense(16, activation='relu'))
#   model.add(layers.Dense(1, activation='sigmoid'))
#	# Compile model
#   model.compile(optimizer='adam'(lr=0.001),
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#   return model
#
### evaluate model with standardized dataset
#estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
###verbose=0 means training progress for each epoch will show you nothing..
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
#print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

















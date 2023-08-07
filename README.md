# Breast_Cancer_Detection
#Importing the Dependencies
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
Data Collection & Processing
# loading and printing the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)
# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns= breast_cancer_dataset.feature_names
# print the first 5 rows of the dataframe
data_frame.head()
# adding the 'target' column to the data frame
data_frame['diagnosis'] = breast_cancer_dataset.target
Getting information about the dataset
# number of rows and columns in the dataset
data_frame.shape
# getting some information about the data
data_frame.info()
# statistical measures about the data
data_frame.describe()
# checking the distribution of Target Varibale       
data_frame['diagnosis'].value_counts()  
   1 357 # 1--Benign
   0 212 # 0--Malignant
   Name: diagnosis, dtype: int64                
“”” We get the mean values of all the features which are group by       diagnosis column”””
data_frame.groupby('diagnosis').mean()

Separating the features and target
# here we dropping the diagnosis column that’s axis value is 1
X = data_frame.drop(columns='diagnosis', axis=1) 
Y = data_frame['diagnosis']
# we can print the X and Y
print(X)
print(Y)
 Splitting the data into training data & Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
         #(569, 30) (455, 30) (114, 30)

Model Training - Logistic Regression
model = LogisticRegression()
# training the Logistic Regression model using Training data
model.fit(X_train, Y_train)
Model Evaluation - Accuracy Score
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data = ', training_data_accuracy)
        #Accuracy on training data =  0.9494505494505494
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data = ', test_data_accuracy)
         #Accuracy on test data =  0.9298245614035088

Building a Predictive System
new_data =(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
new_data_as_numpy_array = np.asarray(new_data)

# reshape the numpy array as we are predicting for one datapoint
new_data_reshaped = new_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(new_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The tumor is Malignant')   #The Malignant tumor is cancerous

else:
  print('The tumor is Benign')      #The Benign tumor is non-cancerous

     [1] The tumor is Benign

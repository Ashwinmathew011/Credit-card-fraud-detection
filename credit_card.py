# importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

# load datasets onto the pandas dataframe
credit_card_data=pd.read_csv(r'E:\Engineering\machineLearning\credit_card_fraud_detection\creditcard.csv')
# dataset info
# #print(credit_card_data.info())

# print the number of missing values in each column

print(credit_card_data.isnull().sum())

# distribution of legit transactions and fraudulent transaction  here in the target class 0 rep valid transaction and 1 rep fraud
print(credit_card_data['Class'].value_counts())

# here we can see that the data is unbalanced where there are 284315 valid and 492 fraud transaction
# 0 --> valid transaction   1 --> fraudulent transaction

# separating data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud=credit_card_data[credit_card_data.Class == 1]

# here in the above statement wherever the class values is 0 itll be stored onto the legit variable
print(legit.shape)

# statistical measure of the data (on amount column)
print(legit.Amount.describe())

# now you'll get a decription of these values which has count(total number of leh=git transaction),mean(mean value)
# std( standard deviation),min(minimum value),then we have percentile values.25% means 25 percentile values are less than the the values displayed

print(fraud.Amount.describe())
# one insight we get from these statistical data is the mean value is considerably higher in fraudelent transaction
# when compared to legit transaction

# compare the values of both transactions -- we use the groupby function on the target 'class' and group them based on the mean of the values
m=credit_card_data.groupby('Class').mean()
print(m)

# we need to deal with the unbalanced data . we use a method known as under sampling.
# build a sample dataset containing legit transactions and fraudulent trasactions
# number of fraudulent transactions is --> 492
# so what is to be done here is randomly pick 492 from the legit tranaction(total --284315) and join it with the frudulent
# transaction. then we get a uniform dataset.

# UNDERSAMPLING

legit_sample=legit.sample(n=492)
# sample returns a random sample of itemsafrom an axis of object, the legit has only the values of class 0. separated from class 1.

# concatenating two dataframes
new_dataset= pd.concat([legit_sample,fraud],axis=0)

# we use the concate function from pandas and join the randomly generated 492 values from legit and the the 492 values from the fraud
# transactions and its concatenated in axis 0 , -- which means the data frame is added one by one(row wise)

print(new_dataset.head())
# head is usedto view the first 5 rows from the dataset

# now lets see the value counts of the newly prepared dataset
print(new_dataset['Class'].value_counts())

new_m=new_dataset.groupby('Class').mean()
print(new_m)
# here the difference is still there between the fraud and legit transaction which is well and good and is most required to
# differentiate between the two.

# after this we can do this we can split our dataset into features and target.
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']

# Split the data into training data and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

# what we are doing here is features are present in X and labels in Y
# we split the X and Y int training data and testing data 
# stratify=Y if we mention this the distribution of Y is evenly distributed amongst the X_train and X_test and randomstate is how you want to run
# the code

print(X.shape,X_train.shape,X_test.shape)
# above statement shows that it has split into test and train data

# MODEL TRAINING USING LOGISTIC REGRESSION
#  this is a binary classification problem so generally we use Logistic regression

model = LogisticRegression()
# the above statement means we are loading one intance of logistic regression onto the model variable
# next comes traing the logistic regression model with training data
model.fit(X_train,Y_train)

# Model evaluation based on ACCURACY SCORE
X_train_prediction= model.predict(X_train) #used to predict the training data for X_train and all those values will be stored in X_train_prediction
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
# we then compare the values predicted to the orginal values to get the accuracy_score
print('Accuracy on training data: ', training_data_accuracy)

# Now we got the accuracy score on the training data and is found to be 94 %
# we calculate the value of training data because if the accuracy is far apart then the model has been overfitted or under fitted
# Now we evaluate the model with the test data that is the data which it has not seen before

# ACCURACY ON TEST DATA

X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print('Accuracy on test data: ', testing_data_accuracy)

cm=confusion_matrix(Y_test,X_test_prediction)
print('\n')
print(cm)
print(classification_report(Y_test,X_test_prediction))

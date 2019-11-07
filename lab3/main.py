from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas
import numpy as np 

# Method to import the data from the dataset 
def importIris():
    
    iris = datasets.load_iris()

    return iris

# Method to split the data into 3 subsets
def splitData(iris):
    
    # Get the 
    X=iris.data
    y=iris.target

    # Split the data into two subsets test 50% and train 50% 
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
    
    print(X_train)
    print("")
    print(y_train)
    print("")
    
    # Now split the test data into two subsets test 25% and validataion 25%
    X_test,X_valid,y_test,y_valid=train_test_split(X_test,y_test, test_size=0.5)
    
    # Now we have 3 subsets of the data
    # Train 50% of data
    # Test 25% of data
    # Validation 25% of data

    print(X_test)
    print("")
    print(y_test)
    print("")
    print(X_valid)
    print("")
    print(y_valid)
    return X_train, y_train, X_test, y_test, X_valid, y_valid


def main():

    iris = importIris()
    print(type(iris))
    irisSubsets = splitData(iris)
    

main()

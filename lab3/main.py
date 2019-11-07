from sklearn import datasets, svm
import matplotlib as plt
from sklearn.model_selection import train_test_split
import pandas
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Step 1 - Import the data from the dataset 
def importIris():
    
    iris = datasets.load_iris()

    return iris

# Step 2 - Split the data into 3 subsets
def splitData(iris):
    
    # Get the 
    X=iris.data
    y=iris.target

    # Split the data into two subsets test 50% and train 50% 
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
    
    print(X_train)
    #print("")
    #print(y_train)
    #print("")
    
    # Now split the test data into two subsets test 25% and validataion 25%
    X_test,X_valid,y_test,y_valid=train_test_split(X_test,y_test, test_size=0.5)
    
    # Now we have 3 subsets of the data
    # Train 50% of data
    # Test 25% of data
    # Validation 25% of data

    #print(X_test)
    #print("")
    #print(y_test)
    #print("")
    #print(X_valid)
    #print("")
    #print(y_valid)
    return X_train, y_train, X_test, y_test, X_valid, y_valid

# Step 3 - Ensure subsets are independent
def subsetTest(irisSubsets, iris):
    
    print(iris.target)
    print("")
    # y_train
    print(irisSubsets[1])
    print("")
    # y_test
    print(irisSubsets[3])
    print("")
    # y_valid
    print(irisSubsets[5])
    print("")
    
    # From visual inspection we see that the subsets are arranged in random order, as the original dataset was
    # Ordered from 0-2 this suggests that the data has been selected randomly
    
    values, counts = np.unique(iris.target, return_counts=True)
    print("Main Data")
    print(values)
    print(counts)
    print("")
    
    print("y_train")
    values, counts = np.unique(irisSubsets[1], return_counts=True)
    print(values)
    print(counts)
    print("")
    
    print("y_test")
    values, counts = np.unique(irisSubsets[3], return_counts=True)
    print(values)
    print(counts)
    print("")

    print("y_valid")
    values, counts = np.unique(irisSubsets[5], return_counts=True)
    print(values)
    print(counts)
    
    # When we count the number of each class in the subsets we see two thing. One the data amount of each 
    # class in a subset always changes on re running the script, meaning that the data is selected randomly
    # and Two the number of each class in each subset will always add up to 50 which was the number of each class
    # in the data set. This implies that there is no overlap in the data. This also shows that the variability
    # of the set if covered and thus is representative 


# Step 4 - building a classifer with SVM
def buildFirstClssifier(irisSubsets):
    
    # Create the linear Support Vector Classifier
    clf = svm.LinearSVC(random_state=0, tol=1e-5)

    # Train it with the train subset
    clf.fit(irisSubsets[0], irisSubsets[1])
    
    print("\n----------SVM----------\n")
    # Print results
    #print("Coef")
    #print(clf.coef_)
    #print("Intercept")
    #print(clf.intercept_)
    #print("")
    print("Test")
    # Test classification with test subset
    prediction_test = clf.predict(irisSubsets[2])
    print(classification_report(irisSubsets[3], prediction_test))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[3], prediction_test))
    print("")
    print("Validataion")
    # Test again with the validation subset
    prediction_valid = clf.predict(irisSubsets[4])
    print(classification_report(irisSubsets[5], prediction_valid))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[5], prediction_valid))


# Step 5 - Building a classifier with Decision Trees
def buildSecondClassifier(irisSubsets):
    
    # Create the Desision tree classifier
    clf = DecisionTreeClassifier(random_state=0)

    # Train it with the data
    clf.fit(irisSubsets[0], irisSubsets[1])
    
    print("\n----------Decision Tree----------\n")
    # Print results
    print("Test")
    # Test classification with test subset
    prediction_test = clf.predict(irisSubsets[2])
    print(classification_report(irisSubsets[3], prediction_test))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[3], prediction_test))
    print("")
    print("Validataion")
    # Test again with the validation subset
    prediction_valid = clf.predict(irisSubsets[4])
    print(classification_report(irisSubsets[5], prediction_valid))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[5], prediction_valid))



# Step 6 - Building a classifier with Logistic regression
def buildThirdClassifier(irisSubsets):
    
    # Create the logisitic regressoin classifier
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    
    tep 7. Select the best out of the three classifiers. (2 points)
    clf.fit(irisSubsets[0], irisSubsets[1])

    print("\n----------Logistic Regression----------\n")
    # Print results
    print("Test")
    # Test classification with test subset
    prediction_test = clf.predict(irisSubsets[2])
    print(classification_report(irisSubsets[3], prediction_test))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[3], prediction_test))
    print("")
    print("Validataion")
    # Test again with the validation subset
    prediction_valid = clf.predict(irisSubsets[4])
    print(classification_report(irisSubsets[5], prediction_valid))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[5], prediction_valid))



# Step 7 - Selecting the best classifier
# Based on the Accuracy socres and classification reports I believe that logistic regression is the best
# classifier. It achieved the highest accuracy score most of the time compared to the other two classifiers.

def main():

    iris = importIris()
    
    irisSubsets = splitData(iris)
    
    subsetTest(irisSubsets, iris)

    buildFirstClssifier(irisSubsets)
    
    buildSecondClassifier(irisSubsets)

    buildThirdClassifier(irisSubsets)

main()

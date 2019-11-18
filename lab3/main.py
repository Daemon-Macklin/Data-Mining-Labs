import pandas
import numpy as np
from sklearn import datasets, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# irisSubsets is a list of the subsets
# i = subset
# 0 = X_train
# 1 = y_train
# 2 = X_test
# 3 = y_test
# 4 = X_valid
# 5 = y_valid

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
    # Add a seed so that the data will be split the same way each time this is run
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5, random_state=7999)
    
    print(X_train)
    #print("")
    #print(y_train)
    #print("")
    
    # Now split the test data into two subsets test 25% and validataion 25%
    # Add a seed so that the data will be split the same way each time this is run
    X_test,X_valid,y_test,y_valid=train_test_split(X_test,y_test, test_size=0.5, random_state=6287)
    
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
    # Return a list of the subsets
    return X_train, y_train, X_test, y_test, X_valid, y_valid

# Step 3 - Ensure subsets are independent
def subsetTest(irisSubsets, iris):
    
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
    # in the data set. This implies that there is no overlap in the data. 
    
    # Also from using the train test split method we can say that the data is independent automatically.
    """
    Main Data
    [0 1 2]
    [50 50 50]

    y_train
    [0 1 2]
    [23 27 25]

    y_test
    [0 1 2]
    [14  9 14]

    y_valid
    [0 1 2]
    [13 14 11]
    """
    
    # Based on the ratos between the amounts of data in the subsets we can say that the data in the subsets
    # Is representative of the actual data
    


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
    
    # Test classification with test subset. Used to check which is best
    prediction_test = clf.predict(irisSubsets[2])
    print(classification_report(irisSubsets[3], prediction_test))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[3], prediction_test))
    print("")
    print("Validataion")
    
    # Validate  with the validation subset. Used if this is best classifier to evalaute future performance 
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
    
    # Test classification with test subset. Used to check which is best
    prediction_test = clf.predict(irisSubsets[2])
    print(classification_report(irisSubsets[3], prediction_test))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[3], prediction_test))
    print("")
    print("Validataion")
    
    # Validate with the validation subset. Used if this is best classifier to evalaute future performance
    prediction_valid = clf.predict(irisSubsets[4])
    print(classification_report(irisSubsets[5], prediction_valid))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[5], prediction_valid))



# Step 6 - Building a classifier with Logistic regression
def buildThirdClassifier(irisSubsets):
    
    # Create the logisitic regressoin classifier
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    
    clf.fit(irisSubsets[0], irisSubsets[1])

    print("\n----------Logistic Regression----------\n")
    # Print results
    print("Test")
    
    # Test classification with test subset. Used to check which is best
    prediction_test = clf.predict(irisSubsets[2])
    print(classification_report(irisSubsets[3], prediction_test))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[3], prediction_test))
    print("")
    print("Validataion")
    
    # Validate again with the validation subset. Used if this is best classifier to evalaute future performance
    prediction_valid = clf.predict(irisSubsets[4])
    print(classification_report(irisSubsets[5], prediction_valid))
    print("Accuracy Score:")
    print(accuracy_score(irisSubsets[5], prediction_valid))



# Step 7 - Selecting the best classifier
"""
----------Logistic Regression----------

Test
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        14
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        14

    accuracy                           1.00        37
   macro avg       1.00      1.00      1.00        37
weighted avg       1.00      1.00      1.00        37

Accuracy Score:
1.0

Based on the Accuracy socres and classification reports I believe that logistic regression is the best
classifier.
"""

# Step 8 - Future Performance on Logistic regression classifier
"""
----------Logistic Regression----------

Validataion
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       0.93      1.00      0.97        14
           2       1.00      0.91      0.95        11

    accuracy                           0.97        38
   macro avg       0.98      0.97      0.97        38
weighted avg       0.98      0.97      0.97        38

Accuracy Score:
0.9736842105263158

After testing with the validation data set the logistic regression classifier has a .97 accuracy score. This 
suggests that it will perform well in future. It is less than the test accuarcy but it is still strong and
thus suggests that the classification will remain accurate on other data sets, as well as proving the test
accuaracy of 1.0 is valid since the classifier remains accurate with new data but still having some errors. 
"""

def main():

    iris = importIris()
    
    irisSubsets = splitData(iris)
    
    subsetTest(irisSubsets, iris)

    buildFirstClssifier(irisSubsets)
    
    buildSecondClassifier(irisSubsets)

    buildThirdClassifier(irisSubsets)

main()

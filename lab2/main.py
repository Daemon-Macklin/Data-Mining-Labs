import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

def importData():
    # Importing CSV file using pandas read_csv function this will give me the converted csv file as a dataframe.
    data = pd.read_csv("BSCY4_Lab_2.csv")
    
    # Print the data 
    # print(data)
    
    # Return data to main function
    return data    


def normalityMPGAssessment(data):
    
    # Use visual inspection as first step of Normaility Testing
    sns.distplot(data["mpg"], kde=True, rug=True)
    plt.title("MPG")
    plt.show()
    
    # The plot is unclear, the distribution is almost normal some other analysis is needed
    stats.shapiro(data["mpg"])
    # Using the Shapiro-Wilk Test we can determine if the data is normal
    # (0.9797349572181702, 0.24196530878543854)
    # The p-value (right) is greater than .05 thus we can say that the data is normal
    # To completely confirm that the data is normal we can perform a Quintile-Quintile Plot
    
    stats.probplot(data["mpg"], dist="norm", plot=plt)
    plt.title("MPG")
    plt.show()
    # This graph is much more clear, backed up by the Shapiro-Wilk test we can say that it is Normal 
    # If the data was not normal we could use a Exponential Transformation to make the data normal. 


def normalityAssessment(data):
    
    # Numertic Fields:
    # Displacement, Acceleration, horse power, weight, mpg(Already done)

    seriesNames = ["displacement", "horsepower", "weight", "acceleration"]
    
    normalityCheckHelper(data, seriesNames)
    #Results:

    # Displacement has a right skewed distribution and with a p-value of 0.00011616637493716553 we can say that the
    # data is not normal. The Q-Q graph also confirms this.
    
    # Horse Power has a bimodal distribution and with a p-value of 0.00016959250206127763 we can say that this data 
    # is not normal. The Q-Q graph also confirms this.

    # Weight seems to have a normal distribution upon inital view. But a p-value of 0.020406033843755722 shows that 
    # this data is not normal. This is also confirmed by the Q-Q graph

    # Acceleration also seems to have a normal distribution upon inital view. But Acceleration has a p-value of
    # 0.5289148092269897 which means that the data is normal. the Q-Q graph will also back this up as there are
    # many more points on the line than the other three.

    # So to try and transform the Displacement, Horse Power and Weight fields, we can transform them and see if
    # they can help us

    # We can make the weight field normal if we use the log transformation.
    data["weight"] = data["weight"].apply(np.log)
    
    normalityCheckHelper(data, ["weight"])
    # Now we get a p-value of 0.20337925851345062 which is normal, and graphs which backup it's normality
    
    # data["displacement"] = (data["displacement"] - data["displacement"].min()) / (data["displacement"].max() - data["displacement"].min())
    data["displacement"] = np.sqrt(data["displacement"])
    print(data["displacement"].describe())
    #data["displacement"] = .5 * np.log(data["displacement"])
    #print(data["displacement"].describe())
    normalityCheckHelper(data, ["displacement"])

    model = sm.OLS(data["mpg"], data["displacement"])
    results = model.fit()
    print(results.summary())
        

def regressionAssumptions(data):
    
    # The Regression Assumptions:
    # 1. Predictor Independence
    # 2. Normality of Predictors and outcome
    # 3. Homoscedasticity of Residuals
    # 4. Lin. Dependency Between Predictors and Outcome.

    # Acceleration is the only Normal Predictor. And so all of the other fields do not fit the other assumptions.
    # So we can only test acceleration.
    model = sm.OLS(data["mpg"], data["acceleration"])
    results = model.fit()
    print(results.summary())
    
    """
                                         OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                    mpg   R-squared (uncentered):                   0.967
    Model:                            OLS   Adj. R-squared (uncentered):              0.966
    Method:                 Least Squares   F-statistic:                              2255.
    Date:                Wed, 16 Oct 2019   Prob (F-statistic):                    2.52e-59
    Time:                        22:37:37   Log-Likelihood:                         -249.26
    No. Observations:                  79   AIC:                                      500.5
    Df Residuals:                      78   BIC:                                      502.9
    Df Model:                           1                                                  
    Covariance Type:            nonrobust                                                  
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    acceleration     1.8739      0.039     47.491      0.000       1.795       1.952
    ==============================================================================
    Omnibus:                        4.718   Durbin-Watson:                   1.133
    Prob(Omnibus):                  0.095   Jarque-Bera (JB):                4.100
    Skew:                           0.401   Prob(JB):                        0.129
    Kurtosis:                       3.777   Cond. No.                         1.00
    ==============================================================================

    """
    plt.figure()
    plt.scatter(data["mpg"], results.resid)
    plt.show()
    
    # After fitting a model for mpg based on acceleration we can see that the model is not valid.
    # As the R-squared value is above .95. This means that there is correlation between the two variables
    # this breaks the Predictor Independence rule and so we cannot use this as a predictor. Also the variance, based
    # On the plot is not homogenos.
    # Currently we cannot use any of the numeric fields as predictors 


def initialRegression(data):
    
    model = sm.OLS(data["mpg"], data)
    results = model.fit()
    print(results.summary())



def main():
    data = importData()
    
    # normalityMPGAssessment(data)

    normalityAssessment(data)

    # regressionAssumptions(data)

    # initialRegression(data)

# Fucntion that will perfrom three step check for normality given the data frame and a list of series to check
def normalityCheckHelper(data, seriesNames):
    
    for series in seriesNames:
        # Distribution curve:
        sns.distplot(data[series], kde=True, rug=True)
        plt.title(series)
        plt.show()

        # Perform Shapiro test
        shapTest = stats.shapiro(data[series])
        print(str(series) + ": " + str(shapTest))

        # Do Q-Q plot to confirm results
        stats.probplot(data[series], dist="norm", plot=plt)
        plt.title(series)
        plt.show()
    

main()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Step 1
def importData():
    # Importing CSV file using pandas read_csv function this will give me the converted csv file as a dataframe.
    data = pd.read_csv("BSCY4_Lab_2.csv")
    
    # Print the data 
    # print(data)
    
    # Return data to main function
    return data    

# Step 2
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
    # If the data was not normal we could use a Log Transformation to make the data normal. 


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
    
    # If we try the same on displacement and horse power they are not normal. However we can find the ratio 
    # between them, and get the expnent of that we end up with a normal series
    data["dis-hp-ratio"] = data["displacement"] / data["horsepower"]
    data["dis-hp-ratio"] = data["dis-hp-ratio"].apply(np.exp)
    normalityCheckHelper(data, ["dis-hp-ratio"])
    # Now we get a p-value of 0.07251033931970596 we can say this data is normal. The histogram and Q-Q plot
    # Also back this up. We will forget about this as we only need two numeric values.
    
    return data

# Step 4 and Step 5
def regressionAssumptions(data):
    
    # The Regression Assumptions:
    # 1. Predictor Independence
    # 2. Normality of Predictors and outcome
    # 3. Homoscedasticity of Residuals
    # 4. Lin. Dependency Between Predictors and Outcome.
    
    #--------------------Step 4--------------------#
    # Acceleration and Weight are the the only Normal Predictor. And so all of the other fields do not fit the other 
    # assumptions.
    
    # First we can test the correlation between our predictors using the pearson method
    print(data["acceleration"].corr(data["weight"]))
    # Weight and Acceleration have a correlation of -0.5593202022603644
    # This is below .95 so there is predictor independance. And we can use the two for our regression model.
    
    #--------------------Step 5--------------------#
    # Now we will build our regression model and ensure it satisifies all of the regression assumptions
    # Build a model with acceleration
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
    plt.title("Acceleration")
    plt.show()
    
    # After fitting a model for mpg based on acceleration we can see that the model is valid.
    # There is Homoscedasticity in the scatter plot. Most of the data points "clump" around 0. 
    # So this does pass the assumptions.

    # Build a model with weight(after a log transformation)
    model = sm.OLS(data["mpg"], data["weight"])
    results = model.fit()
    print(results.summary())

    """
                                         OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                    mpg   R-squared (uncentered):                   0.958
    Model:                            OLS   Adj. R-squared (uncentered):              0.957
    Method:                 Least Squares   F-statistic:                              1767.
    Date:                Thu, 17 Oct 2019   Prob (F-statistic):                    2.40e-55
    Time:                        22:15:57   Log-Likelihood:                         -258.54
    No. Observations:                  79   AIC:                                      519.1
    Df Residuals:                      78   BIC:                                      521.4
    Df Model:                           1                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    weight         3.9473      0.094     42.037      0.000       3.760       4.134
    ==============================================================================
    Omnibus:                        0.573   Durbin-Watson:                   0.958
    Prob(Omnibus):                  0.751   Jarque-Bera (JB):                0.678
    Skew:                          -0.027   Prob(JB):                        0.712
    Kurtosis:                       2.549   Cond. No.                         1.00
    ==============================================================================

    """
    plt.figure()
    plt.scatter(data["mpg"], results.resid)
    plt.title("Weight")
    plt.show()

    # Looking at this graph we can see that there is no Homscedasticity of residuals. As the residuals
    # change as the input changes. And there is no "clump" of data points around 0.
    # Weight does not pass the assumptions of regression

# Step 6
def weightAccelerationRegression(data):
    
    # Since weight an acceleration are not correlated, we can use them both for a regression model.

    X = np.column_stack((data["acceleration"], data["weight"]))
    X = sm.add_constant(X)

    model = sm.OLS(data["mpg"], X)
    results = model.fit()

    print(results.summary())

    """
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    mpg   R-squared:                       0.330
    Model:                            OLS   Adj. R-squared:                  0.312
    Method:                 Least Squares   F-statistic:                     18.70
    Date:                Tue, 22 Oct 2019   Prob (F-statistic):           2.48e-07
    Time:                        15:53:28   Log-Likelihood:                -238.51
    No. Observations:                  79   AIC:                             483.0
    Df Residuals:                      76   BIC:                             490.1
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        187.7695     40.839      4.598      0.000     106.432     269.107
    x1             0.3856      0.353      1.093      0.278      -0.317       1.088
    x2           -21.2526      4.856     -4.377      0.000     -30.923     -11.582
    ==============================================================================
    Omnibus:                        2.342   Durbin-Watson:                   0.731
    Prob(Omnibus):                  0.310   Jarque-Bera (JB):                1.780
    Skew:                           0.355   Prob(JB):                        0.411
    Kurtosis:                       3.189   Cond. No.                     1.30e+03
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.3e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    """

    plt.figure()
    plt.scatter(data["mpg"], results.resid)
    plt.title("Weight + Acceleration")
    plt.show()
    
    # After ploting the graph we can see that there is Homscedasticity in the scatter plot. It is not as 
    # accurate as acceleration. 

# Step 7
def mediationAnalysis(data):
    
    # We compare this to the regression results of mpg based on acceleration, mpg based on weight and mpg based
    # on weight and acceleration. 

    X = sm.add_constant(data["weight"])
    model = sm.OLS(data["acceleration"], X)
    results = model.fit()
    print("Predictor and Modiator Model")
    print(results.summary())
    """
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           acceleration   R-squared:                       0.313
    Model:                            OLS   Adj. R-squared:                  0.304
    Method:                 Least Squares   F-statistic:                     35.06
    Date:                Tue, 22 Oct 2019   Prob (F-statistic):           8.41e-08
    Time:                        19:58:10   Log-Likelihood:                -149.73
    No. Observations:                  79   AIC:                             303.5
    Df Residuals:                      77   BIC:                             308.2
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         75.3979     10.005      7.536      0.000      55.476      95.320
    weight        -7.6959      1.300     -5.921      0.000     -10.284      -5.108
    ==============================================================================
    Omnibus:                        2.119   Durbin-Watson:                   2.122
    Prob(Omnibus):                  0.347   Jarque-Bera (JB):                1.621
    Skew:                           0.167   Prob(JB):                        0.445
    Kurtosis:                       2.383   Cond. No.                         427.
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    """
    # From this we see that when weight is added there is a decline in the R-Squared value
    # from 0.967 to 0.330, and an increase of the p-value of accleration from 0.000 to 0.278. This shows there is
    # an mediation effect and we should drop weight. 
    
# Step 8
def catRegression(data):    

    dummies = pd.get_dummies(pd.Series(data["model year"]))
    X = np.column_stack((data["acceleration"], dummies))
    X = sm.add_constant(X)

    model = sm.OLS(data["mpg"], X)
    results = model.fit()

    print(results.summary())
    """
        ==============================================================================
    Dep. Variable:                    mpg   R-squared:                       0.633
    Model:                            OLS   Adj. R-squared:                  0.560
    Method:                 Least Squares   F-statistic:                     8.642
    Date:                Tue, 22 Oct 2019   Prob (F-statistic):           7.36e-10
    Time:                        21:07:28   Log-Likelihood:                -214.67
    No. Observations:                  79   AIC:                             457.3
    Df Residuals:                      65   BIC:                             490.5
    Df Model:                          13                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          7.9067      3.709      2.132      0.037       0.499      15.315
    x1             1.2689      0.245      5.181      0.000       0.780       1.758
    x2            -1.1225      2.691     -0.417      0.678      -6.496       4.251
    x3             0.8155      1.962      0.416      0.679      -3.104       4.735
    x4            -3.2473      1.746     -1.860      0.067      -6.735       0.240
    x5            -7.7328      1.940     -3.987      0.000     -11.606      -3.859
    x6            -0.9901      1.731     -0.572      0.569      -4.448       2.468
    x7            -0.7086      1.949     -0.364      0.717      -4.601       3.184
    x8            -0.9382      1.971     -0.476      0.636      -4.875       2.999
    x9            -1.1938      1.636     -0.730      0.468      -4.461       2.074
    x10            1.2251      1.441      0.850      0.398      -1.653       4.103
    x11            3.2187      2.743      1.174      0.245      -2.259       8.696
    x12            7.1719      1.175      6.105      0.000       4.826       9.518
    x13            4.5171      1.221      3.699      0.000       2.078       6.956
    x14            6.8917      1.356      5.083      0.000       4.184       9.599
    ==============================================================================
    Omnibus:                        2.750   Durbin-Watson:                   2.105
    Prob(Omnibus):                  0.253   Jarque-Bera (JB):                2.007
    Skew:                           0.310   Prob(JB):                        0.367
    Kurtosis:                       3.475   Cond. No.                     2.24e+17
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 4.21e-31. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    """
    
    plt.figure()
    plt.scatter(data["mpg"], results.resid)
    plt.title("Acceleration + Model Year")
    plt.show()
    
    # Based on the plot the we can say that the model is homogeneous. The regression is valid but based on the 
    # R-Squared value it is not accurate.

    # I think that Model year is the best option as origin is all the same, and Cylinders only has 3 unique
    # catagories. You could use Cylinders but it would not be an accurate regression model

# Step 9
def mediationCategorical(data):

    # To check is there is a mediation effect we shuold compare the regression results between acceleration, model year
    # and acceleration and model year.
    dummies = pd.get_dummies(pd.Series(data["model year"]), drop_first=True)
    X = sm.add_constant(dummies)
    model = sm.OLS(data["mpg"], X)
    results = model.fit()

    print(results.summary())
    """
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    mpg   R-squared:                       0.482
    Model:                            OLS   Adj. R-squared:                  0.388
    Method:                 Least Squares   F-statistic:                     5.120
    Date:                Tue, 22 Oct 2019   Prob (F-statistic):           5.72e-06
    Time:                        21:02:30   Log-Likelihood:                -228.33
    No. Observations:                  79   AIC:                             482.7
    Df Residuals:                      66   BIC:                             513.5
    Df Model:                          12                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         25.5000      3.369      7.569      0.000      18.774      32.226
    71             4.0000      4.126      0.969      0.336      -4.238      12.238
    72            -1.3000      3.986     -0.326      0.745      -9.259       6.659
    73            -5.5000      4.126     -1.333      0.187     -13.738       2.738
    74             3.8333      3.890      0.985      0.328      -3.933      11.600
    75             2.0000      4.126      0.485      0.629      -6.238      10.238
    76             2.5000      4.126      0.606      0.547      -5.738      10.738
    77             1.9167      3.890      0.493      0.624      -5.850       9.683
    78             4.1875      3.767      1.112      0.270      -3.333      11.708
    79             7.4500      4.764      1.564      0.123      -2.062      16.962
    80             9.9000      3.619      2.736      0.008       2.675      17.125
    81             7.4583      3.639      2.050      0.044       0.193      14.724
    82             9.3889      3.724      2.521      0.014       1.953      16.825
    ==============================================================================
    Omnibus:                        0.507   Durbin-Watson:                   1.569
    Prob(Omnibus):                  0.776   Jarque-Bera (JB):                0.172
    Skew:                          -0.090   Prob(JB):                        0.918
    Kurtosis:                       3.141   Cond. No.                         24.1
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    """
    plt.figure()
    plt.scatter(data["mpg"], results.resid)
    plt.title("Model Year")
    plt.show()

    # Based on the change in R-Squared we can say that there is a mediation effect.
    # The model should be updated by removing the catagories in Model Year which have a p-value of
    # above .05 as these are not relevant to the regression model.

def main():
    data = importData()
    
    normalityMPGAssessment(data)

    data = normalityAssessment(data)

    regressionAssumptions(data)

    weightAccelerationRegression(data)

    mediationAnalysis(data)
    
    catRegression(data)
    
    mediationCategorical(data)

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

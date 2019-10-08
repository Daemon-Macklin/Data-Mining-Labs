import pandas as pd
import numpy as np
import pymysql

# STEP 1: Function used to import the csv file and return a data frame
def importCSV():

    # Importing CSV file using pandas read_csv function this will give me the converted csv file as a dataframe.
    data = pd.read_csv("BSCY4.csv")
    
    # Print the data 
    #print(data.columns)
    
    # Return data to main function
    return data

# STEP 2: Functin used to clean date fields
def cleanseDateFields(dataFrame):
    
    # Convert all of the date data points to datetime
    # If there are errors use coerce and change them to NaT
    # Use "infer_datatime_format=True" so that it will be able to read the valid options
    # There are three formats YYYY-MM-DD DD-MM-YYYY and DD/MM. The first two are valid but the dates in DD/MM format
    # are converted to NaT.
    # Using visual inspection I found that:
    # YYYY-MM-DD = 8787
    # DD-MM-YYY = 169
    # MM-YY = 168

    dates = pd.to_datetime(dataFrame["Date"],errors="coerce", infer_datetime_format=True)
    dates = dates.dropna()
    dataFrame["Date"] = dates
    return dataFrame

# STEP 3: Function used to clean type fields
def cleanseTypeFields(typeData):
    
    # Get all of the different types with the unique function ['conventional' 'organic' 'Org.']
    # Clearly Org and organic are meant to be the same so there are 2 valid types. To prevent this to happening again
    # you could use ids corresponding to the type. This would reduce spelling erros and be cleaner.
    
    # Using this function you can find out the number of different types. Just change "Org." to conventional or organic to find the other values
    # Org. = 169 So there are 169 entires that have errors.
    # conventional = 2
    # organic = 8954 
    #print(len(typeData[typeData == 'Org.']))
 
    # Change all of the org items to organic
    typeData[typeData.str.contains("Org.", regex=True)] = "organic"
    #print(typeData.unique())
    return typeData

# STEP 4: Function used to clean average price fields
def cleanseAveragePriceFields(averagePrice):
    
    # Cound the missing values = 20
    #print(averagePrice.isnull().sum())
    
    # Drop the null values 
    averagePrice = averagePrice.dropna() 
    
    # Find the number of items that have , instead of . 
    #print(len(averagePrice[averagePrice.str.contains(",", regex=True)]))
    # 30 items in the data has a , instead of a . this is the erroneous string-based representation.
    # So we must replace them with all with .
    
    averagePrice = averagePrice.str.replace(",",".")

    # Now we can cast all of the data to be numeric

    averagePrice = pd.to_numeric(averagePrice, errors="coerce")
    
    # Drop other null/nan values
    averagePrice = averagePrice.dropna()

    # print(averagePrice.unique())

    return averagePrice

# STEP 6 Function to import data from mysql
def importSQL():
    
    # Connect to the local database
    connection = pymysql.connect(host="localhost", user="dataMiner", password="dataMiner", db="BSCY4")
    
    # Query to get all data from avocado
    query = "SELECT * FROM AVOCADO"
    
    # Execute command and return dataframe
    dataFrame = pd.read_sql(query, connection)
    
    # Close the db connection
    connection.close()
    
    # Print the dataFrame
    #print(dataFrame)
    
    # Return the dataframe
    return dataFrame

# STEP 7: Function to clean region field
def cleanseRegionField(region):
    
    region.dropna()
    # There are 57 Different regions.
    #print("Number of different regions: " + str(len(region.unique())))
   
    # The issue with the region variable is that there is no specific format. Some of the entires are the names of the towns. But others contain the name of the town and the state in the same field and in no particular order. To imporve this data there should be seperate fields for the city, state country, etc.
    
    # Count the errors
    # error1 = len(region[region == "Baltimore-Washington"])
    # There are two different versions of the BaltimoreWashingtion value so we need to make these the same 
    region[region.str.contains("Baltimore-Washington", regex=True)] = "BaltimoreWashington"
    
    # Count the errors
    # error2 = len(region[region.str.contains(" ")])
    
    # There are a total of 149 errors in this field
    #print(str(error0 + error1 + error2))
    
    # There are three different denver fields. Two that have extra spaces so we can remove them with:
    region = region.str.replace(" ", "")

    #print(region.unique())

    return region

# STEP 8: Function to clean year field
def cleanseYearField(year):
    
    # The year field has the years 2015-2018. But saves them in two different formats YYYY and YY.
    # Count the number of items with invalid values 
    error1 = len(year[year == 17])
    error2 = len(year[year == 18])
    
    # There are about 3208 fields with invalid data
    #print("Total Errors in year field = " + str(error1 + error2))

    # Change all of the years represented by YY i.e. 17 and add the 20. so all of the data is YYYY
    year.replace([17, 18], [2017, 2018], inplace=True) 
    
    #print(year.unique())    

    return year

# STEP 9: Function to clean type from the SQL field.
def cleanseSQLTypeField(types):
    
    # There are two versions of the content in the type field. [conventional and Conventional]
    # Drop the null values and then replace all of the Conventional values with conventional as the ones from the csv is conventional
    types.dropna()
    
    # There are 169 errors in this field
    # print("Number of errors in SQL types " + str(len(types[types == "Conventional"])))
    types[types.str.contains("Conventional", regex=True)] = "conventional"
    
    # print(types.unique())

    return types

# STEP 10 + 11: Function to fix errors in visual inspection and to consolidate data
def consolidateData(csvFrame, sqlFrame):

    
    #The "Unnamed: 0" Column is only in the csvFrame. We don't need to do anything about this now. We can decide to keep
    #the field by using an inner(discard) or an outter(keep) merge or concat.
    #In step 2 we convert the dates in the csv data to panadas dates. This adds hh:mm:ss to the data. To keep the data 
    #consistant, we should do the same here.
    

    sqlFrame["Date"] = pd.to_datetime(sqlFrame["Date"],errors="coerce", infer_datetime_format=True)
    sqlFrame["Date"].dropna()
       
    #The column names are different between the two data sets we will also need to change the values so they are the 
    #same. I have chosen to change the csv varialbes to align with the sql variable
    # Set the renamed columns df to be a new one as the function does not work when they are the same
    # i.e csvFrame = csvFrame.rename.....
    
    test = csvFrame.rename(columns={"4046":"c4046", "4225": "c4225", "4770": "c4770", "Total Bags":"TotalBags", "Small Bags" : "SmallBags", "Large Bags":"LargeBags", "XLarge Bags":"XLargeBags"}, errors="raise")
    
    # Set the new df back to be the old one
    csvFrame = test
    
    
    # There are two fields Total Volume and TotalValue. 
    # They could be the same value but we should look at the numbers and see if they a similar.
    #print(csvFrame["Total Volume"].describe())
    #print(sqlFrame["TotalValue"].describe())
    
    """
    CSVFRAME
    count    9.125000e+03
    mean     4.800141e+04
    std      1.429706e+05
    min      8.456000e+01
    25%      4.780870e+03
    50%      1.083858e+04
    75%      3.009600e+04
    max      1.814930e+06
    Name: Total Volume, dtype: float64
    
    SQLFRAME
    count    9.124000e+03
    mean     1.653375e+06
    std      4.748399e+06
    min      3.369970e+04
    25%      1.988675e+05
    50%      4.082880e+05
    75%      1.031132e+06
    max      6.250560e+07
    Name: TotalValue, dtype: float64
    """
    # These number are too different to say that they are the same value so I will not rename the column 
    # to keep them seperate
    
            
    #print(csvFrame.columns)
    #print(sqlFrame.columns)
    
    # To consolidate the data I will use an outter concat along the index axis. This will add all of the columns to 
    # the new data frame. As the data such as Total Volume and Total Value can still be useful even though they
    # are not present in half of the database
    finalFrame = pd.concat([csvFrame, sqlFrame], sort=False)
   
    # Write the data frame to a csv file.
    finalFrame.to_csv("./finalFrame.csv")


def main():
    
    # Call import csv function
    mainCSVDataFrame = importCSV()

    # Call function to clean date field 
    mainCSVDataFrame = cleanseDateFields(mainCSVDataFrame)

    # Call function to clean type field
    mainCSVDataFrame["type"] = cleanseTypeFields(mainCSVDataFrame["type"])

    # Call function to clean average price field
    mainCSVDataFrame["AveragePrice"] = cleanseAveragePriceFields(mainCSVDataFrame["AveragePrice"])
    
    # Import data from mysql
    mainSQLDataFrame = importSQL()

    # Call function to clean region field
    mainSQLDataFrame["region"] = cleanseRegionField(mainSQLDataFrame["region"])
    
    # Call function to clean year field
    mainSQLDataFrame["year"] = cleanseYearField(mainSQLDataFrame["year"])

    # Call function to clean type field
    mainSQLDataFrame["type"] = cleanseSQLTypeField(mainSQLDataFrame["type"])
    
    # Call function to consolidate data
    consolidateData(mainCSVDataFrame, mainSQLDataFrame)

main()

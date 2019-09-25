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
    # There are three formats YYYY-MM-DD DD-MM-YYYY and DD/MM the first two are valid but the dates in DD/MM format
    # are converted to NaT.
    dates = pd.to_datetime(dataFrame["Date"],errors="coerce", infer_datetime_format=True)
    dates = dates.dropna()
    dataFrame["Date"] = dates
    return dataFrame

# STEP 3: Function used to clean type fields
def cleanseTypeFields(typeData):
    
    # Get all of the different types with the unique function ['conventional' 'organic' 'Org.']
    # Clearly Org and organic are meant to be the same so there are 2 valid types
    
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
    # print(averagePrice.unique())

# STEP 6 Function to import data from mysql
def importSQL():
    
    # Connect to the local database
    connection = pymysql.connect(host="localhost", user="dataMiner", password="dataMiner", db="BSCY4")
    
    # Create a cursor
    cursor = connection.cursor()
    
    # Execute the select all from the avocado table  command 
    cursor.execute("select * from AVOCADO")

    # Get all of the data the query returned
    rawData = cursor.fetchall()

    # Put the data into a dataframe
    dataFrame = pd.DataFrame(rawData)
    
    # Rename the column name. As you lose the headings when you take the data from the database. There is a slight difference in the data. The CSV has a field "Unnamed: 0" that is not present in the sql database.
    dataFrame = dataFrame.rename(columns={0:"Date", 1: "AveragePrice", 2: "Total Volume", 3: "4046", 4: "4225", 5: "4770", 6: "Total Bags", 7: "Small Bags", 8: "Large Bags", 9: "XLarge Bags", 10: "type", 11: "year", 12: "region" })
    
    # Print the dataFrame
    #print(dataFrame)
    
    # Return the dataframe
    return dataFrame

# STEP 7: Function to clean region field
def cleanseRegionField(region):
    
    region.dropna()
    # There are 57 Different regions.
    #print("Number of different regions: " + str(len(region.unique())))
   
    # The issue with the region variable is that there is not specific enough. Some of the entires are the names of the towns. But others contain the name of the town and the state in the same field and in no particular order. To imporve this data there should be seperate fields for the city, state country, etc.
    
    # There are two different versions of the BaltimoreWashingtion value so we need to make these the same 
    region[region.str.contains("Baltimore-Washington", regex=True)] = "BaltimoreWashington"
    
    #print(region.unique())

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

# STEP 10 + 11: Function to consolidate data
def consolidateData(csvFrame, sqlFrame):
    
    """
    Visual Inspection
    MAKE A PLOT 
    From visual inspection we can see that the csvFrame has a column that is not present in the sql frame. the 
    The "Unnamed: 0" Column is only in the csvFrame. We don't need to do anything about this now. We can decide to keep
    the field by using an inner(discard) or an outter(keep) merge or concat.

        The other issue is that there will be 2 instances of the same index number.  
    """
    


    """
    Consolidation
    """
    # print(csvFrame)
    # print(sqlFrame)
    
    # To consolidate the data I will use an outter concat along the index axis. This will add the 
    finalFrame = pd.concat([csvFrame, sqlFrame], sort=False)
    
    print(finalFrame.describe())

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
    finalDataFrame = consolidateData(mainCSVDataFrame, mainSQLDataFrame)


main()

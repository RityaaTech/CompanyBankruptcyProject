###########################################################################
#
# Project Name : Smart Company Bankruptcy using Machine Learning 
# Author : Ritesh Saptale
# Date : 29/10/26
#
# Description : 
# The objective is to predict whether a company is likely to go bankrupt 
# or not 
# 
############################################################################

# ---------------- IMPORTING MODULES -----------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib
# ----------------------------------------------------------------

def Company_Bankruptcy():

    Border = "="*70
    New_Border = "-"*70

    #####################################################################
    # STEP : BANNER DISPLAY 
    #####################################################################

    print(Border)
    print(" CORPORATE BANKRUPTCY RISK PREDICTION ".center(70,"="))
    print(Border,end="\n\n")

    # -------------------------------------------------------------------
    #
    # STEP 1 : LOAD THE DATASET
    # Author : Ritesh Saptale 
    # Description : Load the Dataset from csv
    # 
    # -------------------------------------------------------------------

    print(Border)
    print(" STEP 1 : LOAD THE DATASET ".center(70,"="))
    print(Border,end="\n\n")

    print(New_Border,end="\n")

    Datapath = "data.csv"
    df = pd.read_csv(Datapath)

    print("Dataset loaded Successfully.")

    print(New_Border,end="\n\n")
    print(Border,end="\n\n")

    # -------------------------------------------------------------------
    #
    # STEP 2 : DATA ANALYSIS (EDA)
    # Author : Ritesh Saptale 
    # Description : Performing Data Analysis to clean and correct the data
    # 
    # -------------------------------------------------------------------
    
    print(Border)
    print(" STEP 2 : DATA ANALYSIS (EDA) ".center(70,"="))
    print(Border,end="\n\n")
    
    print(New_Border,end="\n")

    # ------------ Print first 5 records ---------------------------------
    print("Initial Entries from Dataset : ")
    print(df.head())
    print(New_Border,end="\n")
    # --------------------------------------------------------------------

    print(f"Shape of the Dataset : {df.shape}")
    print(New_Border,end="\n")

    print(" Columns Names : ".center(70,"-"))
    for i in list(df.columns):
        print(f"Column Name : {i}")
    print(New_Border,end="\n")

    print("Missing Values per columns : ")
    print(df.isnull().sum())
    print(New_Border,end="\n")

    print("Statistical Report of the Dataset : ")
    print(df.describe())

    print(New_Border,end="\n\n")
    print(Border,end="\n\n")

    # -------------------------------------------------------------------
    #
    # STEP 3 : DECIDE INDEPENDENT VARIABLES AND DEPENDENT VARIABLES
    # Author : Ritesh Saptale 
    # Description : Decide Independent and Dependent Variables to send data 
    # for training and Testing to the model
    # 
    # -------------------------------------------------------------------

    print(Border)
    print(" STEP 3 : DECIDE INDEPENDENT VARIABLES AND DEPENDENT VARIABLES ".center(70,"="))
    print(Border,end="\n\n")
        
    print(New_Border,end="\n")
    
    # --------------------------------------------------------------------
    # Features : Independent Variables (X)
    # Labels : Dependent Variables (Y)
    # -------------------------------------------------------------------

    # -------------------- DESCRIPTION ----------------------------------
    # 1) ROA(C) : before interest and depreciation before interest
    # 2) ROA(A) : before interest and % after tax
    # 3) ROA(B) : before interest and depreciation after tax
    # 4) Operating Gross Margin
    # 5) Realized Sales Gross Margin
    # 6) Operating Profit Rate
    # 7) Pre-tax net Interest Rate
    # 8) After-tax net Interest Rate
    # 9) Non-industry income and expenditure/revenue
    # 10) Continuous interest rate (after tax)
    # 11) Operating Expense Rate
    # 12) Research and development expense rate
    # 13) Cash flow rate
    # 14) Interest-bearing debt interest rate
    # 15) Tax rate (A)
    # 16) Net Value Per Share (B)
    # 17) Net Value Per Share (A)
    # 18) Net Value Per Share (C)
    # 19) Persistent EPS in the Last Four Seasons
    # 20) Cash Flow Per Share
    # 21) Revenue Per Share (Yuan Â¥)
    # 22) Operating Profit Per Share (Yuan Â¥)
    # 23) Per Share Net profit before tax (Yuan Â¥)
    # 24) Realized Sales Gross Profit Growth Rate
    # 25) Operating Profit Growth Rate
    # 26) After-tax Net Profit Growth Rate
    # 27) Regular Net Profit Growth Rate
    # 28) Continuous Net Profit Growth Rate
    # 29) Total Asset Growth Rate
    # 30) Net Value Growth Rate
    # 31) Total Asset Return Growth Rate Ratio
    # 32) Cash Reinvestment %
    # 33) Current Ratio
    # 34) Quick Ratio
    # 35) Interest Expense Ratio
    # 36) Total debt/Total net worth
    # 37) Debt ratio %
    # 38) Net worth/Assets
    # 39) Long-term fund suitability ratio (A)
    # 40) Borrowing dependency
    # 41) Contingent liabilities/Net worth
    # 42) Operating profit/Paid-in capital
    # 43) Net profit before tax/Paid-in capital
    # 44) Inventory and accounts receivable/Net value
    # 45) Total Asset Turnover
    # 46) Accounts Receivable Turnover
    # 47) Average Collection Days
    # 48) Inventory Turnover Rate (times)
    # 49) Fixed Assets Turnover Frequency
    # 50) Net Worth Turnover Rate (times)
    # 51) Revenue per person
    # 52) Operating profit per person
    # 53) Allocation rate per person
    # 53) Working Capital to Total Assets
    # 54) Quick Assets/Total Assets
    # 55) Current Assets/Total Assets
    # 56) Cash/Total Assets
    # 57) Quick Assets/Current Liability
    # 58) Cash/Current Liability
    # 59) Current Liability to Assets
    # 60) Operating Funds to Liability
    # 61) Inventory/Working Capital
    # 62) Inventory/Current Liability
    # 63) Current Liabilities/Liability
    # 64) Working Capital/Equity
    # 65) Current Liabilities/Equity
    # 66) Long-term Liability to Current Assets
    # 67) Retained Earnings to Total Assets
    # 68) Total income/Total expense
    # 69) Total expense/Assets
    # 70) Current Asset Turnover Rate
    # 71) Quick Asset Turnover Rate
    # 72) Working capitcal Turnover Rate
    # 73) Cash Turnover Rate
    # 74) Cash Flow to Sales
    # 75) Fixed Assets to Assets
    # 76) Current Liability to Liability
    # 77) Current Liability to Equity
    # 78) Equity to Long-term Liability
    # 79) Cash Flow to Total Assets
    # 80) Cash Flow to Liability
    # 81) CFO to Assets
    # 82) Cash Flow to Equity
    # 83) Current Liability to Current Assets
    # 84) Liability-Assets Flag
    # 85) Net Income to Total Assets
    # 86) Total assets to GNP price
    # 87) No-credit Interval
    # 88) Gross Profit to Sales
    # 89) Net Income to Stockholder's Equity
    # 90) Liability to Equity
    # 91)  Degree of Financial Leverage (DFL)
    # 92)  Interest Coverage Ratio (Interest expense to EBIT)
    # 93)  Net Income Flag
    # 94)  Equity to Liability
    #
    # Target : 
    # 0 : company is likely to go bankrupt
    # 1 : company is not likely to go bankrupt
    # ------------------------------------------------------------------- 
    
    Features_Columns = [
        " Total assets to GNP price",      
        " Debt ratio %",                      
        " Revenue Per Share (Yuan ¥)",     
        " Operating Profit Rate",          
        " Net Income to Total Assets",     
        " Cash/Total Assets"               
    ]

    # 1) Total Assets
    # 2) Total Liabilities
    # 3) Current Assets
    # 4) Current Liabilities
    # 5) Total Revenue
    # 6) Operating Profit (EBIT)
    # 7) Net Income (Net Profit)
    # 8) Cash & Cash Equivalents

    Features = df[Features_Columns]
    Labels = df["Bankrupt?"]

    print(f"Shape of the Features : {Features.shape}")
    print(f"Shape of the Labels : {Labels.shape}")

    print("Decided Independent and Dependent variables successfully.")

    print(New_Border,end="\n\n")
    print(Border,end="\n\n")
    
    # -------------------------------------------------------------------
    #
    # STEP 4 : SPLIT THE DATASET FOR TRAINING AND TESTING
    # Author : Ritesh Saptale 
    # Description : 
    # Split the Dataset for the Training and Testing which will given to
    # the model further
    #
    # -------------------------------------------------------------------
    
    print(Border)
    print(" STEP 4 : SPLIT THE DATASET FOR TRAINING AND TESTING ".center(70,"="))
    print(Border,end="\n\n")
            
    print(New_Border,end="\n")

    Features_train , Features_test , Labels_train , Labels_test = train_test_split(
        Features,
        Labels,
        test_size=0.2,
        random_state=42
    )

    print(f"Shape of Features_train : {Features_train.shape}")
    print(f"Shape of Features_test : {Features_test.shape}")
    print(f"Shape of Labeles_train : {Labels_train.shape}")
    print(f"Shape of Labeles_test : {Labels_test.shape}")
    
    print("Splitted the Dataset for training and testing successfully.")

    print(New_Border,end="\n\n")
    print(Border,end="\n\n")

    # -------------------------------------------------------------------
    #
    # STEP 5 : SCALE THE FEATURES  
    # Author : Ritesh Saptale 
    # Description : 
    # Scale the Features using StandardScaler()
    #
    # -------------------------------------------------------------------

    print(Border)
    print(" STEP 5 : SCALE THE FEATURES ".center(70,"="))
    print(Border,end="\n\n")
    print(New_Border,end="\n")
    
    Scaler = StandardScaler()
    
    Features_train = Scaler.fit_transform(Features_train)
    Features_test = Scaler.fit_transform(Features_test)
    
    print("Features Scaled Successfully.")
    
    print(New_Border,end="\n\n")
    print(Border,end="\n\n")
   
    # -------------------------------------------------------------------
    #
    # STEP 6 : BUILD THE MODEL  
    # Author : Ritesh Saptale 
    # Description : 
    # we are going to Build the Model and we are using 
    # LogisticRegression() model (Best used for binary classification)
    #
    # -------------------------------------------------------------------
    
    print(Border)
    print(" STEP 6 : BUILD THE MODEL ".center(70,"="))
    print(Border,end="\n\n")

    print(New_Border,end="\n")

    Model = LogisticRegression(max_iter=1000)
    print("Model Created Successfully.")

    print(New_Border,end="\n\n")
    print(Border,end="\n\n")

    # -------------------------------------------------------------------
    #
    # STEP 7 : TRAIN THE MODEL  
    # Author : Ritesh Saptale 
    # Description : 
    # We will train the model which we have build by giving the data to it
    #
    # -------------------------------------------------------------------

    print(Border)
    print(" STEP 7 : TRAIN THE MODEL ".center(70,"="))
    print(Border,end="\n\n")
    
    print(New_Border,end="\n")

    Model = Model.fit(Features_train,Labels_train)
    print("Model trained Successfully.")

    print(New_Border,end="\n\n")
    print(Border,end="\n\n")

    # -------------------------------------------------------------------
    #
    # STEP 8 : TEST THE MODEL  
    # Author : Ritesh Saptale 
    # Description : 
    # We will test the model by giving the data to it
    #
    # -------------------------------------------------------------------
    
    print(Border)
    print(" STEP 8 : TEST THE MODEL ".center(70,"="))
    print(Border,end="\n\n")
    
    print(New_Border,end="\n")
    
    Labels_Prediction = Model.predict(Features_test)
    print("Model Tested Successfully.")

    print(New_Border,end="\n\n")
    print(Border,end="\n\n")

    # -------------------------------------------------------------------
    #
    # STEP 9 : EVALUATE THE MODEL  
    # Author : Ritesh Saptale 
    # Description : 
    # We will evaluate the model by checking accuracy scores
    #
    # -------------------------------------------------------------------

    print(Border)
    print(" STEP 9 : EVALUATE THE MODEL ".center(70,"="))
    print(Border,end="\n\n")
       
    print(New_Border,end="\n")

    Accuracy = accuracy_score(Labels_test,Labels_Prediction)
    print(f"Accuracy of the Model 🎯 : {Accuracy*100} %")
    print(New_Border,end="\n")

    print(" Confusion Matrix of the Model ".center(70,"-"))
    Matrix = confusion_matrix(Labels_test,Labels_Prediction)
    print(Matrix)
    print(New_Border,end="\n") 

    print(" Classification Report 📊 ".center(70,"-"))
    Report = classification_report(Labels_test,Labels_Prediction)
    print(Report)
    
    print(New_Border,end="\n\n")
    print(Border,end="\n\n")

    # =====================================================================
    #
    # Step 10 : Ask the Question from the user 
    # Author : Ritesh Saptale
    # Description :  
    # Ask the Question from the user to check whether company is likely 
    # to go bankrupt or not 
    #
    # =====================================================================

    print(Border)
    print(" STEP 10 : TAKE INPUT FROM THE USER ".center(70,"="))
    print(Border,end="\n\n")
    print(New_Border,end="\n")

    Total_assets_to_GNP_price = float(input("Enter the Total Assets : "))
    Debt_ratio = float(input("Enter Total Liabilities (in %) : "))
    CurrentAssets_TotalAssets = float(input("Enter the Current Assets : "))
    Current_Liability_to_Assets = float(input("Enter the Current Liability to Assets : "))
    Revenue_Per_Share = float(input("Enter the Total Revenue : "))
    Operating_Profit_Rate = float(input("Enter the Operating Profit (EBIT) : "))
    NetIncome_to_TotalAssets = float(input("Enter the Net Income (Net Profit) : "))
    Cash_or_TotalAssets = float(input("Enter the Cash & Cash Equivalents : "))

    User = pd.DataFrame({
            "Total Assets" : [Total_assets_to_GNP_price],
            "Total Liabilities (in %)" : [Debt_ratio],
            "Total Revenue" : [Revenue_Per_Share],
            "Operating Profit (EBIT)" : [Operating_Profit_Rate],
            "Net Income (Net Profit)" : [NetIncome_to_TotalAssets],
            "Cash & Cash Equivalents" : [Cash_or_TotalAssets]
        })

    Prediction = Model.predict(User)
    
    if(Prediction[0] == 1):
        print("Company is not likely to get Bankruptcy 📈")
    else:
        print("Company is not likely to get Bankruptcy 📉")
    
    print(New_Border,end="\n\n")
    print(Border,end="\n\n")

    # =====================================================================
    #
    # Step 11 : Preserve the Model 
    # Author : Ritesh Saptale
    # Description :  
    # Preserve the Model for the further use case 
    #
    # =====================================================================
    
    print(Border)
    print(" STEP 11 : PRESERVE THE MODEL ".center(70,"="))
    print(Border,end="\n\n")
    print(New_Border,end="\n")
    
    joblib.dump(Model,"CompanyBankruptcy01.pkl")
    print("Model saved successfully.")
    
    print(New_Border,end="\n\n")
    print(Border,end="\n\n")

    #####################################################################
    # STEP : BANNER DISPLAY 
    #####################################################################
    
    print(Border)
    print(" END OF THE APPLICATION ".center(70,"="))
    print(Border,end="\n\n")
    
def main():
    Company_Bankruptcy()

if __name__ == "__main__":
    main()

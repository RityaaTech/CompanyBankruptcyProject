SMART COMPANY BANKRUPTCY PREDICTION
====================================

Author: Ritesh Saptale
Project Date: 29/10/2026

1. PROJECT OVERVIEW
-------------------
This project uses Machine Learning to predict whether a company is likely
to go bankrupt based on selected financial indicators.

The project contains:
- Main.py: Loads the dataset, performs basic data analysis, selects features,
  trains and evaluates a Logistic Regression model, accepts a sample user
  input, and saves the trained model as CompanyBankruptcy.pkl.
- CompanyBankruptcyGUI.py: A Streamlit dashboard that loads the saved model
  and displays a bankruptcy prediction.
- data.csv: The dataset required by Main.py.
- CompanyBankruptcy.pkl: The trained model file required by the GUI.
- Requirement.txt: Python package dependencies.

2. INPUT FEATURES
-----------------
The training script selects these eight dataset columns:
1. Total assets to GNP price
2. Debt ratio %
3. Current Assets/Total Assets
4. Current Liability to Assets
5. Revenue Per Share (Yuan)
6. Operating Profit Rate
7. Net Income to Total Assets
8. Cash/Total Assets

Target column:
- Bankrupt?

Target labels in Main.py:
- 0: Company is likely to go bankrupt.
- 1: Company is not likely to go bankrupt.

The Streamlit dashboard presents corresponding company-financial input
fields to the user.

3. REQUIREMENTS
---------------
- Python 3
- pandas
- numpy
- scikit-learn
- joblib
- streamlit

Install dependencies from the project folder:

    py -m pip install -r Requirement.txt

4. PROJECT FOLDER
-----------------
Keep the required files together in one folder, for example:

    CompanyBankruptcyProject/
        Main.py
        CompanyBankruptcyGUI.py
        data.csv
        CompanyBankruptcy.pkl
        Requirement.txt
        Readme.txt

The dataset and trained model are generated/provided separately; they are
not included automatically by the Python scripts.

5. TRAIN THE MACHINE LEARNING MODEL
-----------------------------------
1. Open Command Prompt or PowerShell.
2. Change directory to the project folder.
3. Ensure data.csv is present in that folder.
4. Run:

    py Main.py

The script reads data.csv, trains and evaluates a Logistic Regression
classifier, asks for a sample company input, and saves the model as:

    CompanyBankruptcy.pkl

6. RUN THE STREAMLIT WEB APPLICATION
------------------------------------
Ensure CompanyBankruptcyGUI.py and CompanyBankruptcy.pkl are in the same
folder. Then run:

    py -m streamlit run CompanyBankruptcyGUI.py

Streamlit will print a local URL in the terminal. Open that URL in a
browser to use the dashboard.

7. IMPORTANT MODEL PREPROCESSING NOTE
-------------------------------------
Main.py scales the training features using StandardScaler before fitting
the Logistic Regression model. However, the current script saves only the
Logistic Regression model, not the fitted scaler. CompanyBankruptcyGUI.py
currently sends the entered values directly to the model without applying
that scaler.

For predictions to be consistent with training, the fitted scaler should
also be saved in Main.py and loaded/applied in CompanyBankruptcyGUI.py
before calling Model.predict(). The GUI's input column names and order
should also match the training features.

Until this preprocessing is made consistent, predictions from the GUI
may be unreliable. This project is an educational machine-learning
demonstration, not professional financial advice.

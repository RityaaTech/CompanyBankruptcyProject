import pandas as pd
import streamlit as st
import joblib


def CompanyBankruptcyGUI():
    # ---------------- CUSTOM DASHBOARD THEME ----------------

    st.markdown("""
        <style>

        /* Main Background */
        .stApp {
            background: linear-gradient(
                135deg,
                #0f172a,
                #172554,
                #312e81
            );
            color: white;
        }

        /* Main Title */
        h1, h2, h3 {
            color: #60a5fa !important;
            font-weight: bold;
        }

        /* Normal Text */
        p, label {
            color: #e2e8f0 !important;
        }

        /* Input Boxes */
        div[data-baseweb="input"] input {
            background-color: #1e293b;
            color: white;
            border: 1px solid #3b82f6;
            border-radius: 8px;
        }

        /* Predict Button */
        div.stButton > button {
            background: linear-gradient(
                90deg,
                #2563eb,
                #7c3aed
            );
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 25px;
            font-weight: bold;
            width: 100%;
        }

        div.stButton > button:hover {
            background: linear-gradient(
                90deg,
                #7c3aed,
                #2563eb
            );
            color: white;
            border: 1px solid white;
        }

        /* Divider */
        hr {
            border-color: #475569;
        }

    """, unsafe_allow_html=True)
    # ---------------------------------------------------------

    st.header("FinGuard AI 🏢📈")

    st.write("FinGuard AI is the Application that predicts whether a company is financially at risk of bankruptcy or not based on its financial indicators.")

    Model = joblib.load("CompanyBankruptcy01.pkl")

    st.subheader("ENTER THE COMPANY DETAILS  : ")

    Total_assets_to_GNP_price = st.number_input("Enter the Total Assets :")
    st.write("Total Assets: The total value of everything a company owns, including cash, property, equipment, and investments.")

    Debt_ratio = st.number_input("Enter Total Liabilities (in %) : ")
    st.write("Debt Ratio: Measures the proportion of a company's total assets financed through debt.")

    #CurrentAssets_TotalAssets = st.number_input("Enter the Current Assets : ")
    #Current_Liability_to_Assets = st.number_input("Enter the Current Liability to Assets : ")
    Revenue_Per_Share = st.number_input("Enter the Total Revenue : ")
    st.write("Total Revenue: The total income a company earns from selling goods or providing services before deducting expenses.")

    Operating_Profit_Rate = st.number_input("Enter the Operating Profit (EBIT) : ")
    st.write("Operating Profit (EBIT): The profit a company earns from its core business operations before deducting interest and taxes")

    NetIncome_to_TotalAssets = st.number_input("Enter the Net Income (Net Profit) : ")
    st.write("Net Income (Net Profit): The total profit a company earns after deducting all expenses, interest, and taxes from its revenue.")

    Cash_or_TotalAssets = st.number_input("Enter the Cash & Cash Equivalents : ")
    st.write("Cash & Cash Equivalents: The money a company has available in cash or in its bank accounts.")

    if st.button("Predict Bankruptcy 🏛️"):
        New_user = pd.DataFrame({
                   "Total Assets" : [Total_assets_to_GNP_price],
                    "Total Liabilities (in %)" : [Debt_ratio],
                    #"Current Assets" : [CurrentAssets_TotalAssets],
                    #"Current Liability to Assets" : [Current_Liability_to_Assets],
                    "Total Revenue" : [Revenue_Per_Share],
                    "Operating Profit (EBIT)" : [Operating_Profit_Rate],
                    "Net Income (Net Profit)" : [NetIncome_to_TotalAssets],
                    "Cash & Cash Equivalents" : [Cash_or_TotalAssets]
    })

        prediction = Model.predict(New_user)[0]

        if prediction == 1:
            st.success("Company is not likely to get Bankruptcy 📈")
        else:
            st.error("Company is likely to get Bankruptcy 📉.")
            st.error("NOTE ⚠️: Please improve your Company Financials")
                
    st.divider()
    
    st.caption("Author : Ritesh Saptale 👩‍💻")

def main():
    CompanyBankruptcyGUI()

if __name__ == "__main__":
    main()

# Command to Execute : python -m streamlit run CompanyBankruptcyGUI.py
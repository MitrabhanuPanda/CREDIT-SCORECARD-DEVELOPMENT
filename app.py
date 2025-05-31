import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle

model=pickle.load(open("MODEL/pipe.pkl","rb"))
df=pickle.load(open(r"MODEL/df.pkl","rb"))
df1=pd.read_csv(r"C:\Users\mitra\OneDrive\Desktop\CREDIT RISK\DATASET\4. CREDIT SCORECARD\final_data.csv")


st.title("LOAN DEFAULTER MODEL PREDICTION")

per_income=st.number_input("PERSON INCOME",value=None,step=1)
ownership=st.selectbox("HOME OWNERSHIP",["RENT", "MORTGAGE", "OWN", "OTHER"])
loan_percent = st.number_input("PERCENTAGE OF INCOME",value=None, max_value=1.00,min_value=0.00)
int_rat = st.number_input("INTEREST RATE",value=None, max_value=100.00,min_value=0.00)
cb_person_default_on_file = st.selectbox("CREDIT DEFAULT ON FILE",["Y", "N"])



# Set your scorecard parameters
base_score = 600  # By Default

pdo = 20 # Point to Double the Odds

base_odds = 50

# Calculate Factor & Offset

factor = pdo/math.log(2)

offset = base_score + factor * math.log(base_odds)

# Convert Predicted PD to Score
def pd_to_score(pd):
    odds = (1-pd)/pd
    score = offset - factor * np.log(odds)
    
    return score


if st.button("PREDICTION"):

    if ownership=="RENT":
        ownership=0
    elif ownership=="OWN":
        ownership=1
    elif ownership=="MORTGAGE":
        ownership=2
    else:
        ownership=3

    if cb_person_default_on_file=="Y":
        cb_person_default_on_file=1
    else:
        cb_person_default_on_file=0


    new_query=np.array([[per_income,ownership,loan_percent,int_rat,cb_person_default_on_file]])
    new_query=new_query.reshape(1,5)

    prob_pred=model.predict_proba(new_query)[:, 1][0]
    
    credit_scores=pd_to_score(prob_pred)

    display_score= int(credit_scores)

    if display_score<0:
        st.title(f"Your Credit Score is : {0}" )
    elif display_score>900:
        st.title(f"Your Credit Score is : {900}" )
    else:
        st.title(f"Your Credit Score is : {display_score}" )
    
    















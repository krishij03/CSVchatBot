import streamlit as st
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import SmartDataframe
import matplotlib.pyplot as plt

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def chat_with_csv(df, prompt):
    llm = OpenAI(api_token=openai_api_key, model="gpt-4")
    pandas_ai = SmartDataframe(df, config={"llm": llm})

    result = pandas_ai.chat(prompt)
    if isinstance(result, pd.DataFrame):
        return st.table(result)
    else:
        return st.text(result)

st.set_page_config(layout='wide')
st.title("AgriTech: AI Researcher")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)

# Upload multiple CSV files and other interactions under "AgriTech: AI Researcher"
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

if input_csvs:
    selected_file = st.selectbox("", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data, use_container_width=True)

    input_text = st.text_area("Enter the query")
    if input_text and st.button("Chat with csv"):
        result = chat_with_csv(data, input_text)
        if plt.get_fignums():
            st.pyplot(plt.gcf())
        else:
            st.success(result)

st.subheader('Tabular Representations and LLM')
# Power BI iframe under "Tabular Representations and LLM"
powerbi_iframe = """<iframe title="imp" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=df3dc983-e878-4115-8505-52385651c08f&autoAuth=true&ctid=d1f14348-f1b5-4a09-ac99-7ebf213cbc81" frameborder="0" allowFullScreen="true"></iframe>"""
st.markdown(powerbi_iframe, unsafe_allow_html=True)

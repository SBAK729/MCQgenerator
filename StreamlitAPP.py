import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from langchain_community.callbacks.manager import get_openai_callback
import streamlit as st
from src.mcqgenerator.MCQgenerator import generate_evaluation_chain
from src.mcqgenerator.logger import logging

# Loading response json file
with open("D:\DirectEd\Geni AI\MCQgenerator\Response.json","r") as file:
    RESPONSE_JSON = json.load(file)

# Creating a title for the app
st.title("MCQs Creator Application with Langchain")

# Create a form using Streamlit st.form
with st.form("user_input"):
    # File Upload
    uploaded_file = st.file_uploader("Upload pdf or txt file")

    #Input Fields
    mcq_count = st.number_input("No. of MCQs", max_value=50, min_value=3)

    #Subject
    subject = st.text_input("Insert subject", max_chars=20)

    #Quiz Tone
    tone = st.text_input("Complexity leel of Questions", max_chars=20, placeholder="Simple")

    # Add Button
    button = st.form_submit_button("Create MCQS")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading..."):
            try:
                text = read_file(uploaded_file)
                # Count tokens and the cost of API call

                with get_openai_callback() as cb:
                    response = generate_evaluation_chain.invoke(
                        {
                            "text":text,
                            "number":mcq_count,
                            "subject":subject,
                            "tone":tone,
                            "response_json":json.dumps(RESPONSE_JSON)
                        }
                    )

                    # st.write(response)
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")
            
            else:
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost: {cb.total_cost}")

                if isinstance(response, dict):
                    # Extract rhe quiz data from the response
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)

                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)

                            # Display the review in a text box as well
                            st.text_area(label="Review", value=response["revies"])

                        else:
                            st.error("Error in the table data")
                
                else:
                    st.write(response)


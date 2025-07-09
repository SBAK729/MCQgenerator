import os
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file, get_table_data

# Packeages from lang chain

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# Load environment variable from .env

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
os.environ.pop("SSL_CERT_FILE", None)

llm = ChatOpenAI(api_key=key, model_name="gpt-3.5-turbo", temperature=0.7)

template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to\
create a quiz of {number} multiple choise questions for {subject} students in {tone}.
make sure the questions are not repeated and check all the quesions to be conforming the text as wll.
make sure to format your response like RESPONSE_JSON bellow and use it as a guide.\
Ensure to make {number} of MCQs.
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone" , "response_json"],
    template=template
)

quiz_chain = LLMChain(llm = llm, prompt = quiz_generation_prompt, output_key = 'quiz', verbose=True)

template2 = """
You are an expert english grammarian and writer.Given a multiple choice quiz for a {subject} students.\
We need to evaluate the complexity of the questions and give a complete analysis of the quiz.Only use at max 50 words for complaxity if the quiz is at per the with the cognitive and analytical ability of the students.\
Update the quiz questions which needs to be changed and change the tone such that it perfectly fits the students abilities.
Quiz_MCQs:
{quiz}

check from an expert english writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"],template=template2)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt,output_key="review", verbose=True)

generate_evaluation_chain = SequentialChain(chains = [quiz_chain,review_chain], input_variables = ["text", "number", "subject", "tone" , "response_json"], output_variables = ["quiz","review"], verbose = True)
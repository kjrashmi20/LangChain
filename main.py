#Integrate code with Open AI API
import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain

os.environ['OpenAI_API_KEY']=openai_key

#streamlit framework
st.title('Network Technologies Search Results')
input_text = st.text_input("Search")

# Prompt Template
first_input_template = PromptTemplate(input_variables=['name'],template="Tell me about {name}")

#Open AI LLM
llm = OpenAI(temperature = 0.8)
chain = LLMChain(llm=llm,prompt=first_input_template,verbose=True)

if input_text:
    st.write(llm(input_text))

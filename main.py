#Integrate code with Open AI API
import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OpenAI_API_KEY']=openai_key

#streamlit framework
st.title('Network Technologies Search Results')
input_text = st.text_input("Search")

# Prompt Template
first_input_template = PromptTemplate(input_variables=['name'],template="Tell me about {name}")
second_input_template = PromptTemplate(input_variables=['definition'],template='Benefits of {definition}')
third_input_template = PromptTemplate(input_variables=['pros'],template='Latest updates on {pros}')

#Memory
tech_memory = ConversationBufferMemory(input_key='name',memory_key='tech_history')
def_memory = ConversationBufferMemory(input_key='definition',memory_key='def_history')
update_memory = ConversationBufferMemory(input_key='pros',memory_key='desc_history')

#Open AI LLM
llm = OpenAI(temperature = 0.8)
chain = LLMChain(llm=llm,prompt=first_input_template,verbose=True,
output_key='definition',memory=tech_memory)
chain2 = LLMChain(llm=llm,prompt=second_input_template,verbose=True,
output_key='pros',memory=def_memory)
chain3 = LLMChain(llm=llm,prompt=third_input_template,verbose=True,
output_key='description',memory=update_memory)

parent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['definition','pros','description'],verbose=True)


if input_text:
    st.write(parent_chain({'name': input_text}))

    with st.expander('Tech Name'):
        st.info(tech_memory.buffer)
    
    with st.expander('Description'):
        st.info(update_memory.buffer)

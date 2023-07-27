import os
import json
import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
from utils import load_csv
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, validator
from typing import Dict, List

class SuggestedMapping(BaseModel):
    map: Dict[str, List[str]]


# Set up LLM and store in session
if 'chat_model' not in st.session_state.keys():
    OPEN_AI_KEY = os.environ['OPEN_AI_KEY']
    TEMPERATURE = 0
    MODEL = "gpt-3.5-turbo-0613"
    chat_model = ChatOpenAI(
        openai_api_key=OPEN_AI_KEY,
        temperature=TEMPERATURE,
        model=MODEL
    )
    st.session_state.chat_model = chat_model


parser = PydanticOutputParser(pydantic_object=SuggestedMapping)
parser = OutputFixingParser.from_llm(parser=parser, llm=st.session_state.chat_model)

# App title
st.set_page_config(page_title="🤗💬 HugChat")

# Layout
col1, col2 = st.columns(2)
sidebar = st.sidebar

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    content = "I'm Tabby, a helpful AI assistant for organizing your data how you want."
    content += "\nPlease upload a template file in csv format with the desired column names and data formats."
    content += "\nThen upload a source file that you would like converted to the template format."
    st.session_state.messages = [{"role": "assistant", "content": content}]


# Set up memory and store in session
if 'memory' not in st.session_state.keys():
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    st.session_state.memory = memory

# Set up logical gate for column disambiguation
if 'columns_disamb' not in st.session_state.keys():
    st.session_state.columns_disamb = False

if 'col2val' not in st.session_state.keys():
    st.session_state.col2val = {}

if 'template_df' not in st.session_state.keys():
    st.session_state.template_df = None

if 'target' not in st.session_state.keys():
    st.session_state.target = None

if 'target_df' not in st.session_state.keys():
    st.session_state.target_df = None

if 'template' not in st.session_state.keys():
    st.session_state.template = None

if 'tables_processed' not in st.session_state.keys():
    st.session_state.tables_processed = 0


# Display chat messages
for message in st.session_state.messages:
    with sidebar.chat_message(message["role"]):
        sidebar.write(message["content"])


uploader_message = "Upload a template in csv format."
st.session_state.template = col1.file_uploader(uploader_message, key='CSVTemplate')
# Get Template and Target CSV Files
if st.session_state.template is not None:
    try:
        st.session_state.template_df = load_csv(st.session_state.template)
    except Exception as e:
        with sidebar.chat_message("assistant"):
            response = f'Unfortunately, there was an error processing your template file\n{str(e)}'
            response += '\nPlease double check your file and retry the upload'
            sidebar.write(response)
        st.session_state.template_df = None
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


uploader_message = "Upload a source file to convert to the template format"
st.session_state.target = col2.file_uploader(uploader_message, key='CSVTarget')
if st.session_state.target is not None:
    try:
        st.session_state.target_df = load_csv(st.session_state.target)
    except Exception as e:
        with sidebar.chat_message("assistant"):
            response = f'Unfortunately, there was an error processing your source file\n{str(e)}'
            response += '\nPlease double check your file and retry the upload'
            sidebar.write(response)
        st.session_state.target_df = None
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if (
        st.session_state.template_df is not None
        and st.session_state.target_df is not None
        and not st.session_state.tables_processed
):
    with sidebar.chat_message("assistant"):
        with st.spinner("Thank you. Please wait while I process your tables..."):
            col1.dataframe(st.session_state.template_df)
            col2.dataframe(st.session_state.target_df)

            template_columns = list(st.session_state.template_df.columns)
            target_columns = list(st.session_state.target_df.columns)

            tmp = "You are a helpful assistant. Your job is to map columns in the template table to one or more "
            tmp += " columns in the target table. The template table columns are: {template_columns}."
            tmp += "\n{format_instructions}\n"
            tmp += "Here are the target table columns: {target_columns}."
            prompt = PromptTemplate.from_template(tmp)
            input = prompt.format_prompt(
                template_columns=', '.join(template_columns),
                format_instructions=parser.get_format_instructions(),
                target_columns=', '.join(target_columns)
            )
            qa = ConversationChain(llm=st.session_state['chat_model'], memory=st.session_state['memory'])
            response = qa.run(input=input.to_string())
            suggested_mapping = parser.parse(response)
            temp_cols = ', '.join(template_columns)
            targ_cols = ', '.join(target_columns)
        response = f'I found the following columns in the template table:\n{temp_cols}\nAnd I found these columns'
        response += f' in the other table you uploaded:\n{targ_cols}.\n'
        response += f'Based on this information, I would suggest the following possible mappings.'
        response += '\n\nIf there is more than one possible mapping, please choose one in the form below.'
        sidebar.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
        st.session_state.tables_processed = 1


if not st.session_state.columns_disamb:
        with sidebar.chat_message("assistant"):
            with sidebar.form("disambiguate_columns"):

                for col in st.session_state.template_df.columns:
                    choices = suggested_mapping.map[col]
                    if len(choices) > 1:
                        st.session_state.col2val[col] = sidebar.radio(col, choices)
                    else:
                        st.session_state.col2val[col] = choices[0]
                columns_disamb = st.form_submit_button("Submit")
                st.session_state.columns_disamb = columns_disamb
else:
    with sidebar.chat_message('assistant'):
        resp = "Got it. Thank you for choosing the columns. I have the following mapping:\n"
        for col, val in st.session_state.col2val.items():
            print('\t' + col + ': ' + val)



# Function for generating LLM response
def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot                        
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)

# # User-provided prompt
# if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)
#
# # Generate a new response if last message is not from assistant
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = generate_response(prompt, hf_email, hf_pass)
#             st.write(response)
#     message = {"role": "assistant", "content": response}
#     st.session_state.messages.append(message)

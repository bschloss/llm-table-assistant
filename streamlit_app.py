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
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import Dict, List

# App title
st.set_page_config(page_title="🤗💬 HugChat")

# Hugging Face Credentials
with st.sidebar:
    st.title('🤗💬 HugChat')
    if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
        st.success('HuggingFace Login credentials already provided!', icon='✅')
        hf_email = st.secrets['EMAIL']
        hf_pass = st.secrets['PASS']
    else:
        hf_email = st.text_input('Enter E-mail:', type='password')
        hf_pass = st.text_input('Enter password:', type='password')
        if not (hf_email and hf_pass):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    content = "I'm Tabby, a helpful AI assistant for organizing your data how you want."
    content += "\nPlease upload a template file in csv format with the desired column names and data formats."
    st.session_state.messages = [{"role": "assistant", "content": content}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Get Template and Target CSV Files
template = st.file_uploader("Upload a template in csv format.", key='CSVTemplate')
csv_template = None
if template is not None:
    try:
        csv_template = load_csv(template)
        with st.chat_message("assistant"):
            response = 'Thank you!'
            st.write(response)
    except Exception as e:
        with st.chat_message("assistant"):
            response = f'Unfortunately, there was an error processing your file\n{str(e)}'
            response += '\nPlease double check your file and retry the upload'
            st.write(response)
            csv_template = None
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

csv_target = None
if csv_template is not None:
    uploader_message = "Now please upload another CSV file that you would like converted to the template format"
    target = st.file_uploader(uploader_message, key='CSVTarget')
    if target is not None:
        try:
            csv_target = load_csv(target)
            with st.chat_message("assistant"):
                response = 'Thank you!'
                st.write(response)
        except Exception as e:
            with st.chat_message("assistant"):
                response = f'Unfortunately, there was an error processing your file\n{str(e)}'
                response += '\nPlease double check your file and retry the upload'
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if csv_target is not None:
    template_columns = list(csv_template.columns)
    target_columns = list(csv_target.columns)

    # Set up Chat Model and store in session
    OPEN_AI_KEY = os.environ['OPEN_AI_KEY']
    TEMPERATURE = 0
    MODEL = "gpt-3.5-turbo-0613"
    chat_model = ChatOpenAI(
        openai_api_key=OPEN_AI_KEY['OPEN_AI_KEY'],
        temperature=TEMPERATURE,
        model=MODEL
    )
    st.session_state['chat_model'] = chat_model

    # Set up parser and store in session
    class SuggestedMapping(BaseModel):
        map: Dict[str, List[str]]
    parser = PydanticOutputParser(pydantic_object=SuggestedMapping)
    st.session_state['parser'] = parser

    # Set up memory and store in session
    st.session_state['memory'] = ConversationBufferMemory(memory_key="history", return_messages=True)

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
    with st.chat_message("assistant"):
        temp_cols = ', '.join(template_columns)
        targ_cols = ', '.join(target_columns)
        mapping_text = json.dumps(suggested_mapping, indent=4)
        response = f'I found the following columns in the template table:\n{temp_cols}\nAnd I found these columns'
        response += f' in the other table you uploaded:\n{targ_cols}.\n'
        response += f'Based on this information, I would suggest the following mapping:\n{mapping_text}'
        st.write(response)




# Function for generating LLM response
def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot                        
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)

# User-provided prompt
if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, hf_email, hf_pass) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

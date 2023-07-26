import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
from utils import load_csv

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

import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message as st_message

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
MODEL_NAME = 'gpt-4'
TEMPERATURE = 0.1

def main():
    st.set_page_config(page_title="Chat With Any Files")
    st.header("Chatbot")

    initialize_session_state()
    
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        
        # API Key selection
        api_key_option = st.radio("Choose API Key Option:", ("Use Default Key", "Enter My Own Key"))
        
        if api_key_option == "Enter My Own Key":
            openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        else:
            openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
            if not openai_api_key:
                st.warning("Default API key not found in secrets. Please enter your own key.")
                openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        
        process = st.button("Process")

    if process:
        if not uploaded_files:
            st.warning("Please upload at least one file before processing.")
        elif not openai_api_key:
            st.warning("Please provide an OpenAI API key to continue.")
        else:
            st.session_state.processing = True
            st.session_state.chat_history = []  # Clear chat history
            st.session_state.conversation = None  # Reset conversation
            st.session_state.processComplete = False
            process_files(uploaded_files, openai_api_key)
            st.session_state.processing = False

    if st.session_state.processing:
        st.info("Processing files... Please wait.")
    elif st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_user_input(user_question)
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        st_message(message.content, is_user=i % 2 == 0, key=str(i))

def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = ""

def process_files(uploaded_files, openai_api_key):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    
    st.session_state.debug_info = "Starting file processing...\n"
    files_text = get_files_text(uploaded_files)
    if not files_text:
        st.error("No text could be extracted from the uploaded files. Please check the file contents and try again.")
        return

    st.session_state.debug_info += f"Extracted text length: {len(files_text)}\n"
    text_chunks = get_text_chunks(files_text)
    if not text_chunks:
        st.error("No text chunks were created. The extracted text might be too short.")
        return

    st.session_state.debug_info += f"Number of text chunks: {len(text_chunks)}\n"
    try:
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True
        st.success("Files processed successfully!")
        st.session_state.debug_info += "Vectorstore and conversation chain created successfully.\n"
    except Exception as e:
        st.error(f"An error occurred while processing the files: {str(e)}")
        st.session_state.debug_info += f"Error during processing: {str(e)}\n"

    st.expander("Debug Information").text(st.session_state.debug_info)

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        elif file_extension == ".xlsx":
            text += get_excel_text(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
        st.session_state.debug_info += f"Processed file: {uploaded_file.name}\n"
    return text

def get_pdf_text(pdf):
    try:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        st.session_state.debug_info += f"PDF text extracted, length: {len(text)}\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        st.session_state.debug_info += f"Error reading PDF: {str(e)}\n"
        return ""

def get_docx_text(file):
    try:
        doc = docx.Document(file)
        text = ' '.join(para.text for para in doc.paragraphs)
        st.session_state.debug_info += f"DOCX text extracted, length: {len(text)}\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {str(e)}")
        st.session_state.debug_info += f"Error reading DOCX: {str(e)}\n"
        return ""

def get_excel_text(file):
    try:
        # Read Excel file
        xls = pd.ExcelFile(file)
        text = ""

        for sheet_name in xls.sheet_names:
            text += f"\nSheet: {sheet_name}\n"
            
            # Read the entire sheet
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Process the dataframe in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                for _, row in chunk.iterrows():
                    # Convert row to string, handling various data types
                    row_text = ", ".join([f"{col}: {process_cell(val)}" for col, val in row.items()])
                    text += row_text + "\n"
            
            # Add basic sheet statistics
            text += f"\nSheet Statistics:\n"
            text += f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n"
            text += f"Column Names: {', '.join(df.columns)}\n"

        st.session_state.debug_info += f"Excel text extracted, length: {len(text)}\n"
        return text
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        st.session_state.debug_info += f"Error reading Excel: {str(e)}\n"
        return ""

def process_cell(val):
    if pd.isna(val):
        return "N/A"
    elif isinstance(val, datetime):
        return val.strftime('%d %B %Y')
    elif isinstance(val, (int, float)):
        return f"{val:,}"
    else:
        return str(val)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    st.session_state.debug_info += f"Text split into {len(chunks)} chunks\n"
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not text_chunks:
        raise ValueError("No text chunks to process")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    st.session_state.debug_info += f"Vectorstore created with {len(text_chunks)} chunks\n"
    return vectorstore

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=MODEL_NAME, temperature=TEMPERATURE)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    st.session_state.debug_info += "Conversation chain created\n"
    return conversation_chain

def handle_user_input(user_question):
    st.session_state.debug_info += f"User question: {user_question}\n"
    with get_openai_callback() as cb:
        # Use `invoke` instead of `__call__`
        response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.session_state.debug_info += f"AI response generated, tokens used: {cb.total_tokens}\n"

    st.write(f"Total Tokens: {cb.total_tokens}, "
             f"Prompt Tokens: {cb.prompt_tokens}, "
             f"Completion Tokens: {cb.completion_tokens}, "
             f"Total Cost (USD): ${cb.total_cost:.4f}")

    st.expander("Debug Information").text(st.session_state.debug_info)

if __name__ == '__main__':
    main()

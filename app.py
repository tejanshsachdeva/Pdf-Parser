import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message as st_message
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
MODEL_NAME = 'gpt-3.5-turbo'
TEMPERATURE = 0

def main():
    st.set_page_config(page_title="Chat With Any Files")
    st.header("Chatbot")

    initialize_session_state()
    
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'xlsx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not uploaded_files:
            st.warning("Please upload at least one file before processing.")
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
        df_dict = pd.read_excel(file, sheet_name=None)
        text = ""
        for sheet_name, df in df_dict.items():
            text += f"\nSheet: {sheet_name}\n"
            text += df.to_string(index=False) + "\n"
            
            # Add basic statistics
            text += f"\nBasic Statistics for {sheet_name}:\n"
            text += df.describe().to_string() + "\n"
            
            # Add correlation information
            if df.select_dtypes(include=[np.number]).shape[1] > 1:
                text += f"\nCorrelations in {sheet_name}:\n"
                text += df.corr().to_string() + "\n"
            
            # Add information about unique values in each column
            for column in df.columns:
                unique_values = df[column].nunique()
                text += f"\nUnique values in {column}: {unique_values}\n"

        st.session_state.debug_info += f"Excel text extracted, length: {len(text)}\n"
        return text
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        st.session_state.debug_info += f"Error reading Excel: {str(e)}\n"
        return ""

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.session_state.debug_info += f"Text split into {len(chunks)} chunks\n"
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    if not text_chunks:
        raise ValueError("No text chunks to process")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
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
        response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.session_state.debug_info += f"AI response generated, tokens used: {cb.total_tokens}\n"

    st.write(f"Total Tokens: {cb.total_tokens}, "
             f"Prompt Tokens: {cb.prompt_tokens}, "
             f"Completion Tokens: {cb.completion_tokens}, "
             f"Total Cost (USD): ${cb.total_cost:.4f}")

    st.expander("Debug Information").text(st.session_state.debug_info)

if __name__ == '__main__':
    main()

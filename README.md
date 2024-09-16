# PDF Parser Chatbot

## Overview

The **PDF Parser Chatbot** is a web-based application built using **Streamlit** that allows users to upload PDF, DOCX, and Excel files and interact with the content through an AI-powered chatbot. The application uses **OpenAI GPT-4** or a custom API key to generate conversational responses based on the content of the uploaded files. It processes the text, creates embeddings, and utilizes **Langchain** and **FAISS** for conversational retrieval.

## Features

- **File Upload**: Supports multiple file formats - PDF, DOCX, and Excel.
- **Text Extraction**: Extracts text from the uploaded files for processing.
- **Conversational AI**: Chat with the AI about the content of the uploaded files.
- **Embeddings**: Uses HuggingFace embeddings to process chunks of text.
- **Vector Store**: Utilizes FAISS to store and retrieve text chunks efficiently.
- **OpenAI GPT-4 Integration**: Powered by OpenAI's language models to provide responses.

## Setup

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/tejanshsachdeva/Pdf-Parser.git
   cd Pdf-Parser
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables**:

   - Create a `.env` file in the root directory and add your OpenAI API key:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     ```
4. **Run the application**:

   ```bash
   streamlit run app.py
   ```

### Required Libraries

- **Streamlit**: The core framework used to build the web application.
- **PyPDF2**: For reading and extracting text from PDF files.
- **Docx**: For extracting text from DOCX files.
- **Pandas**: Used for processing Excel files.
- **Langchain**: For managing the AI conversation flow and retrieval.
- **FAISS**: For vector-based text search.
- **HuggingFace Transformers**: For text embeddings.
- **Streamlit Chat**: To display chat messages in the UI.

### Environment Setup

Make sure to set up your OpenAI API key in the `.env` file for the app to interact with the OpenAI GPT-4 model. You can also choose to input the key directly in the app.

## Usage

1. **File Upload**: You can upload multiple files (PDF, DOCX, Excel) from the sidebar.
2. **API Key**: Choose to either use the default OpenAI API key (set in secrets) or enter your own API key.
3. **Processing Files**: After uploading files and entering the API key, click the "Process" button to start the extraction.
4. **Ask Questions**: Once the files are processed, use the chat box to ask questions about the contents of your files. The AI will respond with relevant answers based on the text extracted from the uploaded files.
5. **Debug Info**: A debug information panel will help track the steps and issues during the file processing.

## Functions

### `main()`

The main function that handles UI rendering and initializing the chatbot. It handles file upload, API key selection, and user interactions.

### `initialize_session_state()`

Initializes the session state variables that manage the chatbot's memory, chat history, and file processing status.

### `process_files(uploaded_files, openai_api_key)`

Handles the extraction of text from uploaded files, chunking the text, and creating a vector store for retrieval.

### `get_files_text(uploaded_files)`

Extracts the text from the uploaded PDF, DOCX, or Excel files.

### `get_pdf_text(pdf)`

Extracts text from PDF files using the PyPDF2 library.

### `get_docx_text(file)`

Extracts text from DOCX files using the `docx` library.

### `get_excel_text(file)`

Extracts text from Excel files using Pandas and converts it into a readable format.

### `process_cell(val)`

Handles individual cell values in Excel files, converting them into a human-readable format.

### `get_text_chunks(text)`

Splits the extracted text into smaller chunks using a recursive text splitter.

### `get_vectorstore(text_chunks)`

Uses HuggingFace embeddings to convert text chunks into vectors and stores them in FAISS.

### `get_conversation_chain(vectorstore, openai_api_key)`

Creates a conversational retrieval chain using the OpenAI model and vector store for interaction.

### `handle_user_input(user_question)`

Processes the user's question and returns a response using the conversation chain.

---

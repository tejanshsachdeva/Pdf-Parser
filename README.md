**Chat With Any Files**

This application allows users to upload PDF or DOCX files, process them to extract text, and ask questions about the content of those files. The questions are answered using an AI-powered chatbot, either based on OpenAI's GPT-3.5 model or Hugging Face's conversational models.

### Installation

Clone and then Install dependencies:
pip install -r requirements.txt
Few of the dependencies may be outdated and need selective updates. Do debug accordingly.

### Usage

1. Run the application:

   streamlit run main.py

2. Once the application is running, you'll see a Streamlit interface with options to upload files, enter your OpenAI API key, and process the uploaded files.

3. Upload PDF or DOCX files containing text that you want to inquire about.

4. Enter your OpenAI API key in the provided text input field.

5. Click the "Process" button to start the processing of uploaded files.

6. Once the processing is complete, you can ask questions about the uploaded files in the chat interface provided.

### Functionality

- **File Upload**: Users can upload PDF or DOCX files containing text.
- **OpenAI API Integration**: Users can provide their OpenAI API key to utilize the GPT-3.5 model for conversational responses.
- **Text Extraction**: The application extracts text from uploaded files using PyPDF2 (for PDFs) and python-docx (for DOCX files).
- **Text Chunking**: Extracted text is split into chunks to accommodate the input size limitations of the AI model.
- **Conversational AI**: Questions asked by users are answered by the AI-powered chatbot, based on the content of the uploaded files.
- **Feedback and Metrics**: The application provides information such as total tokens used, prompt tokens, completion tokens, and the total cost incurred (if applicable, for OpenAI).

### Contributing

Contributions are welcome! If you have suggestions for improving this application, please open an issue or submit a pull request.
if you want to visit orignal source code of author: https://github.com/sudan94/chat-with-pdf-doc

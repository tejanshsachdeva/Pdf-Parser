# Pdf-Parser

Pdf-Parser is a Streamlit-based application that allows users to upload and analyze various document types, including PDFs, Word documents, and Excel files. It uses natural language processing to answer questions about the uploaded documents.

## Features

- Support for PDF, DOCX, and Excel (.xlsx, .xls) files
- Multiple file upload
- Natural language question answering about the document content
- Conversation history
- Debug information display

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/tejanshsachdeva/Pdf-Parser.git
   cd Pdf-Parser
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Use the sidebar to upload your documents (PDF, DOCX, or Excel files).

4. Ask questions about the uploaded documents in the chat interface.

5. View the conversation history and debug information as needed.

## File Structure

- `app.py`: Main application file containing the Streamlit interface and document processing logic.
- `requirements.txt`: List of Python packages required for the project.

## Dependencies

Main dependencies include:
- streamlit
- PyPDF2
- python-docx
- openpyxl
- pandas
- langchain
- faiss-cpu
- sentence-transformers

For a full list of dependencies, see `requirements.txt`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

Tejansh Sachdeva - [GitHub](https://github.com/tejanshsachdeva)

Project Link: [https://github.com/tejanshsachdeva/Pdf-Parser](https://github.com/tejanshsachdeva/Pdf-Parser)
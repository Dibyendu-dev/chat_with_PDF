# PDF Question Answering with LangChain and Qdrant

This project is a Python-based tool that allows users to ask questions about the content of a PDF document. It uses LangChain for document processing, OpenAI for embeddings, and Qdrant as a vector database for similarity search.

---

## Features

- **PDF Parsing**: Extracts text from a PDF file.
- **Text Splitting**: Splits the extracted text into manageable chunks for processing.
- **Embeddings**: Uses OpenAI's embedding model to generate vector representations of the text.
- **Vector Search**: Stores and retrieves document chunks using Qdrant for similarity-based search.
- **Interactive Q&A**: Allows users to ask questions about the PDF content and get relevant answers.

---

## Prerequisites

1. **Python**: Ensure Python 3.8 or higher is installed.
2. **Docker**: Install Docker Desktop to run Qdrant.
3. **OpenAI API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/).
4. **Dependencies**: Install the required Python libraries.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/chat_with_pdf.git
cd chat_with_pdf
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

5. Start the Qdrant container:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## Usage

1. Place your PDF file in the `data` directory of the project.

2. Run the main script:

```bash
python main.py
```

3. The script will:

   - Process the PDF file
   - Create embeddings
   - Store them in Qdrant
   - Start an interactive session where you can ask questions

4. Example interaction:

```
Enter your question about the PDF: What is the main topic of the document?
```

---

## Project Structure

```
chat_with_pdf/
├── data/               # Directory for PDF files
├── src/               # Source code
│   ├── main.py        # Main application script
│   ├── pdf_processor.py # PDF processing utilities
│   └── qa_engine.py   # Question answering engine
├── .env               # Environment variables
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the document processing framework
- [OpenAI](https://openai.com/) for the embedding models
- [Qdrant](https://qdrant.tech/) for the vector database

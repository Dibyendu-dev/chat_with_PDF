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
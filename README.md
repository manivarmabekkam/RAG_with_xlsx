# RAG_with_xlsx# RAG with Excel Files

A Retrieval-Augmented Generation (RAG) system that works with Excel data using LangChain and Google Vertex AI's Gemini model.

## Overview

This project demonstrates how to build a question-answering system that can retrieve information from Excel files and generate accurate responses using Google's Gemini model. The system uses FAISS for efficient vector similarity search and HuggingFace embeddings for converting text to vector representations.

## Features

- Load and process data from Excel files
- Convert Excel rows to documents with metadata
- Generate embeddings using HuggingFace's sentence transformers
- Create a FAISS vector store for efficient similarity search
- Perform RAG queries using Google Vertex AI's Gemini model
- Retrieve contextually relevant information from Excel data

## Prerequisites

- Python 3.8+
- Google Cloud account with Vertex AI API enabled
- Google Cloud project with appropriate permissions
- Excel file with data (default: `store_inventory.xlsx`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/RAG_with_xlsx.git
   cd RAG_with_xlsx
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Additional dependencies (not in requirements.txt):
   ```bash
   pip install langchain langchain-google-vertexai langchain-community sentence-transformers openpyxl
   ```

## Setup

1. Set up Google Cloud authentication:
   ```bash
   gcloud auth application-default login
   ```

2. Update the project ID and location in the code:
   ```python
   vertexai.init(project="YOUR_PROJECT_ID", location="YOUR_LOCATION")
   ```

3. Prepare your Excel file (default: `store_inventory.xlsx`) with the data you want to query.

## Usage

```python
from agents_gc import rag_query

# Ask a question about your Excel data
answer = rag_query("which product have higher discount in summer?")
print(answer)
```

You can also run the script directly:

```bash
streamlit run agents_gc.py
```

## How It Works

1. **Data Loading**: The system loads data from an Excel file using pandas.
2. **Document Preparation**: Each row from the Excel file is converted to a Document object with metadata.
3. **Embedding Generation**: HuggingFace's sentence transformer model converts text to vector embeddings.
4. **Vector Store Creation**: FAISS creates an efficient vector store for similarity search.
5. **Query Processing**: When a user asks a question, the system:
   - Retrieves the most relevant rows from the Excel file
   - Passes the context and question to Google's Gemini model
   - Returns the generated answer

## Code Structure

```python
# 1. Initialize Vertex AI
vertexai.init(project="agents-458316", location="us-central1")

# 2. Load Excel Data
df = pd.read_excel("store_inventory.xlsx")

# 3. Prepare documents
documents = []
for _, row in df.iterrows():
    text = " ".join(str(v) for v in row.values)
    documents.append(Document(page_content=text, metadata=dict(row)))

# 4. Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Create FAISS vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# 6. Create QA Chain with Vertex AI
llm = VertexAI(
    model_name="gemini-1.5-flash",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
)

# 7. RAG Query Function
def rag_query(user_query):
    response = qa_chain.run(user_query)
    return response
```

## Dependencies

faiss_cpu==1.7.4
google_cloud_aiplatform==1.48.0
langchain==0.3.24
langchain_community==0.3.23
langchain_google_vertexai==2.0.21
numpy==2.2.5
pandas==2.2.3
streamlit==1.34.0
vertexai==1.71.1

## Note

This project requires a Google Cloud account with Vertex AI API enabled and appropriate permissions. Make sure to update the project ID and location in the code before running.

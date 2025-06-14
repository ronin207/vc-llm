# Verifiable Credentials RAG System

This project implements a Retrieval Augmented Generation (RAG) system specifically designed for answering queries about Verifiable Credentials (VCs). The system uses a router-based architecture with RAG-Fusion for enhanced internal document retrieval.

## Features

- Intelligent query routing to determine if questions can be answered from the VC knowledge base
- RAG-Fusion for improved retrieval quality
- Multi-query vector search with Reciprocal Rank Fusion
- Contextualized and accurate answer generation
- Privacy-aware response generation

## Project Structure

```
.
├── src/
│   ├── data/           # Data processing and storage
│   ├── models/         # LLM and embedding models
│   ├── utils/          # Utility functions
│   └── chains/         # LangChain components
├── tests/              # Test files
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

[Usage instructions will be added as the implementation progresses]

## Development

[Development guidelines will be added as the implementation progresses] 
import pytest
from unittest.mock import Mock, patch
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from src.models.router import RouterConfig, DataSource
from src.chains.vc_rag_system import VCRAGSystem
from src.utils.data_processor import VCDataProcessor

@pytest.fixture
def mock_vector_store():
    return Mock(spec=Chroma)

@pytest.fixture
def router_config():
    return RouterConfig(
        model_name="gpt-4",
        temperature=0.0,
        max_tokens=150
    )

@pytest.fixture
def vc_rag_system(mock_vector_store, router_config):
    return VCRAGSystem(
        vector_store=mock_vector_store,
        router_config=router_config
    )

def test_vc_data_processor_json():
    # Test JSON to text conversion
    vc_json = {
        "type": ["VerifiableCredential", "PassportCredential"],
        "issuer": "https://example.com/issuer",
        "issuanceDate": "2023-01-01",
        "expirationDate": "2028-01-01",
        "credentialSubject": {
            "id": "did:example:123",
            "name": "John Doe",
            "nationality": "US"
        }
    }
    
    text = VCDataProcessor.json_to_text(vc_json)
    assert "Types: VerifiableCredential, PassportCredential" in text
    assert "Issuer: https://example.com/issuer" in text
    assert "Issued on: 2023-01-01" in text
    assert "Expires on: 2028-01-01" in text
    assert "name: John Doe" in text
    assert "nationality: US" in text

def test_vc_rag_system_not_found(vc_rag_system):
    # Mock the router to return not_found
    vc_rag_system.router.route = Mock(return_value=Mock(
        datasource=DataSource.NOT_FOUND,
        reasoning="Question is not related to VCs"
    ))
    
    result = vc_rag_system.answer("What is the weather like?")
    
    assert result["source"] == "not_found"
    assert "cannot find the information" in result["answer"].lower()
    assert result["reasoning"] == "Question is not related to VCs"

def test_vc_rag_system_vc_knowledge_base(vc_rag_system):
    # Mock the router to return vc_knowledge_base
    vc_rag_system.router.route = Mock(return_value=Mock(
        datasource=DataSource.VC_KNOWLEDGE_BASE,
        reasoning="Question is about a VC"
    ))
    
    # Mock the RAG-Fusion retrieval
    mock_docs = [
        Mock(page_content="Passport expires on 2028-01-01", metadata={"id": "1"}),
        Mock(page_content="Issued by Example Issuer", metadata={"id": "2"})
    ]
    vc_rag_system.rag_fusion.retrieve = Mock(return_value=mock_docs)
    
    # Mock the generation chain
    vc_rag_system.generation_chain.run = Mock(return_value="The passport expires on January 1, 2028.")
    
    result = vc_rag_system.answer("When does the passport expire?")
    
    assert result["source"] == "vc_knowledge_base"
    assert "passport expires" in result["answer"].lower()
    assert result["reasoning"] == "Question is about a VC"
    assert "Passport expires on 2028-01-01" in result["context"] 
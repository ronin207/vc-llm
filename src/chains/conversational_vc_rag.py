import os
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from models.router import RouterConfig
from utils.data_processor import VCDataProcessor

class ConversationalVCRAG:
    def __init__(
        self,
        vector_store: Chroma,
        router_config: RouterConfig,
        generation_model_name: str = "gpt-4-turbo-preview",
        generation_temperature: float = 0.0,
        memory_key: str = "chat_history",
        allow_private_info: bool = False
    ):
        self.vector_store = vector_store
        self.router_config = router_config
        self.allow_private_info = allow_private_info
        self.last_documents = []
        
        # Initialize the LLM for generation
        self.llm = ChatOpenAI(
            model_name=generation_model_name,
            temperature=generation_temperature
        )
        
        # Initialize memory with output key
        self.memory = ConversationBufferMemory(
            memory_key=memory_key,
            output_key="answer",
            return_messages=True
        )
        
        # Create QA chain
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create a conversational QA chain."""
        # Create the prompt template
        template = """You are a helpful assistant that answers questions about Verifiable Credentials.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        If the question is not about Verifiable Credentials, say that you can only answer questions about Verifiable Credentials.
        
        Context: {context}
        
        Chat History:
        {chat_history}
        
        Human: {question}
        Assistant: Let me help you with that. """
        
        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create the chain with improved retrieval
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5  # Increase number of retrieved documents
                }
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,  # Ensure we get source documents
            return_generated_question=True
        )
    
    def _format_context(self, documents: List) -> str:
        """Format retrieved documents into a string for context."""
        formatted_context = []
        for doc in documents:
            if hasattr(doc, 'metadata') and 'credential_type' in doc.metadata:
                formatted_context.append(
                    f"Credential Type: {doc.metadata['credential_type']}\n"
                    f"Content: {doc.page_content}\n"
                )
        return "\n".join(formatted_context)
    
    def _extract_source_data(self, documents: List) -> List[Dict]:
        """Extract relevant source data from documents."""
        source_data = []
        for doc in documents:
            if hasattr(doc, 'metadata'):
                source_data.append({
                    "credential_type": doc.metadata.get("credential_type", "Unknown"),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        return source_data
    
    def process_query(self, question: str) -> Dict:
        """Process a question and return the answer with context."""
        # Get the answer from the QA chain using invoke
        result = self.qa_chain.invoke({"question": question})
        
        # Store the last retrieved documents
        self.last_documents = result.get("source_documents", [])
        
        # Format the context
        context = self._format_context(self.last_documents)
        
        # Extract source data
        source_data = self._extract_source_data(self.last_documents)
        
        return {
            "answer": result["answer"],
            "reasoning": "The answer was generated based on the retrieved Verifiable Credentials.",
            "source": "vc_knowledge_base" if self.last_documents else "not_found",
            "context": context,
            "source_data": source_data,  # Include the actual source data
            "generated_question": result.get("generated_question", question)
        }
    
    def get_follow_up_suggestions(self) -> List[str]:
        """Generate follow-up questions based on the last context."""
        if not self.last_documents:
            return []
        
        # Create a prompt for generating follow-up questions
        follow_up_prompt = f"""Based on the following Verifiable Credentials information, suggest 3 relevant follow-up questions:
        
        {self._format_context(self.last_documents)}
        
        Generate 3 follow-up questions that would help explore this information further.
        Questions should be specific and related to the credentials shown.
        Return only the questions, one per line."""
        
        # Get follow-up questions from the LLM
        response = self.llm.invoke(follow_up_prompt)
        questions = [q.strip() for q in response.content.split('\n') if q.strip()]
        
        return questions[:3]  # Return at most 3 questions 
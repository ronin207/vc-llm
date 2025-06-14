from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

from models.router import RouterConfig, DataSource
from chains.router_chain import RouterChain
from chains.rag_fusion import RAGFusion

class VCRAGSystem:
    def __init__(
        self,
        vector_store: Chroma,
        router_config: RouterConfig,
        generation_model_name: str = "gpt-4",
        generation_temperature: float = 0.7
    ):
        self.vector_store = vector_store
        
        # Initialize router
        self.router = RouterChain(router_config)
        
        # Initialize RAG-Fusion
        self.rag_fusion = RAGFusion(
            vector_store=vector_store,
            llm=ChatOpenAI(
                model_name=generation_model_name,
                temperature=generation_temperature
            )
        )
        
        # Initialize generation prompt
        self.generation_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an AI assistant that answers questions about Verifiable Credentials.
            Use the following context to answer the question. If the answer cannot be found in the context,
            state that the information is not available in the Verifiable Credentials knowledge base.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        # Create the chain using the new RunnableSequence pattern
        self.generation_chain = self.generation_prompt | ChatOpenAI(
            model_name=generation_model_name,
            temperature=generation_temperature
        )

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:")
            context_parts.append(doc.page_content)
            context_parts.append("---")
        return "\n".join(context_parts)

    def answer(self, question: str) -> Dict[str, Any]:
        """Process a question and generate an answer."""
        # Route the question
        route_result = self.router.route(question)
        
        if route_result["datasource"] == DataSource.NOT_FOUND:
            return {
                "answer": "I apologize, but I cannot find the information you're looking for in the Verifiable Credentials knowledge base.",
                "reasoning": route_result["reasoning"],
                "source": "not_found"
            }
        
        # Retrieve relevant documents using RAG-Fusion
        documents = self.rag_fusion.retrieve(question)
        
        # Format context
        context = self._format_context(documents)
        
        # Generate answer
        response = self.generation_chain.invoke({
            "question": question,
            "context": context
        })
        
        return {
            "answer": response.content,
            "reasoning": route_result["reasoning"],
            "source": "vc_knowledge_base",
            "context": context
        } 
from typing import List, Dict, Any
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class RAGFusion:
    def __init__(
        self,
        vector_store: Chroma,
        llm: ChatOpenAI,
        num_queries: int = 4,
        top_k: int = 3
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.num_queries = num_queries
        self.top_k = top_k
        
        # Initialize query generation prompt
        self.query_generation_prompt = PromptTemplate(
            input_variables=["question", "num_queries"],
            template="""Given the following question, generate {num_queries} different ways to ask the same question.
            Each query should capture a different aspect or perspective of the original question.
            Original question: {question}
            
            Generate {num_queries} queries:
            1. """
        )
        
        # Create the chain using the new RunnableSequence pattern
        self.query_generation_chain = self.query_generation_prompt | self.llm

    def generate_queries(self, question: str) -> List[str]:
        """Generate multiple queries from the original question."""
        response = self.query_generation_chain.invoke({
            "question": question,
            "num_queries": self.num_queries
        })
        
        # Parse the response to extract individual queries
        queries = [q.strip() for q in response.content.split("\n") if q.strip()]
        return queries[:self.num_queries]

    def reciprocal_rank_fusion(self, results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Combine and re-rank results using Reciprocal Rank Fusion."""
        # Initialize scores dictionary
        scores = {}
        
        # Calculate RRF scores
        for rank, result_list in enumerate(results):
            for doc in result_list:
                doc_id = doc.metadata.get("id", str(doc))
                if doc_id not in scores:
                    scores[doc_id] = {"score": 0, "doc": doc}
                scores[doc_id]["score"] += 1.0 / (rank + 1)
        
        # Sort by RRF score
        ranked_docs = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [item["doc"] for item in ranked_docs]

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Perform RAG-Fusion retrieval."""
        # Generate multiple queries
        queries = self.generate_queries(question)
        
        # Retrieve documents for each query
        all_results = []
        for query in queries:
            results = self.vector_store.similarity_search(
                query,
                k=self.top_k
            )
            all_results.append(results)
        
        # Combine and re-rank results
        fused_results = self.reciprocal_rank_fusion(all_results)
        return fused_results[:self.top_k] 
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import json

from models.router import DataSource, RouterConfig

class RouterChain:
    def __init__(self, config: RouterConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Initialize router prompt
        self.router_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a router that determines whether a question can be answered using the Verifiable Credentials knowledge base.
            
            Question: {question}
            
            Determine if this question can be answered using the Verifiable Credentials knowledge base.
            If yes, respond with "vc_knowledge_base". If no, respond with "not_found".
            
            Provide your reasoning for this decision.
            
            Your response must be in the following JSON format:
            {{
                "datasource": "vc_knowledge_base" or "not_found",
                "reasoning": "your reasoning here"
            }}
            
            IMPORTANT: Respond ONLY with the JSON object, no other text.
            """
        )
        
        # Create the chain using the new RunnableSequence pattern
        self.router_chain = self.router_prompt | self.llm

    def route(self, question: str) -> Dict[str, Any]:
        """Route a question to the appropriate data source."""
        try:
            # Get response from LLM
            response = self.router_chain.invoke({"question": question})
            
            # Extract the JSON string from the response
            json_str = response.content.strip()
            
            # Parse the JSON response
            result = json.loads(json_str)
            
            # Validate the datasource
            datasource = DataSource(result["datasource"])
            
            return {
                "datasource": datasource,
                "reasoning": result["reasoning"]
            }
        except Exception as e:
            # If there's any error in parsing, default to not_found
            return {
                "datasource": DataSource.NOT_FOUND,
                "reasoning": f"Error parsing router response: {str(e)}"
            } 
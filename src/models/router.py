from enum import Enum
from pydantic import BaseModel, Field

class DataSource(str, Enum):
    """Enum for possible data sources."""
    VC_KNOWLEDGE_BASE = "vc_knowledge_base"
    NOT_FOUND = "not_found"

class RouteQuery(BaseModel):
    """Model for query routing decision."""
    datasource: DataSource = Field(
        description="The data source that should be used to answer the query"
    )
    reasoning: str = Field(
        description="Explanation of why this data source was chosen"
    )

class RouterConfig(BaseModel):
    """Configuration for the router."""
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 150 
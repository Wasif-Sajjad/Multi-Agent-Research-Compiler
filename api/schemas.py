from pydantic import BaseModel, Field
from typing import List

class ResearchRequest(BaseModel):
    query: str = Field(..., description="The main research question or topic to investigate.")
    max_iterations: int = Field(3, description="Maximum number of critic evaluation loops allowed.", ge=1, le=10)

class ResearchResponse(BaseModel):
    final_report: str = Field(..., description="The synthesized markdown report with inline citations.")
    sources: List[str] = Field(default_factory=list, description="Deduplicated list of sources used in the report.")
    iteration_count: int = Field(..., description="The number of research loops taken to reach the final report.")

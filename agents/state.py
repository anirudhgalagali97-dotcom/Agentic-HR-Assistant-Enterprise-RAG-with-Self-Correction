"""
LangGraph State Definition for Agentic RAG System
Defines the state schema and routing logic
"""
from typing import TypedDict, List, Optional, Literal, Union
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class GradeScore(BaseModel):
    """Grading score for document relevance."""
    binary_score: str = Field(
        description="'yes' if document is relevant to question, 'no' otherwise"
    )
    score: float = Field(
        description="Relevance score between 0 and 1"
    )
    reasoning: str = Field(
        description="Reasoning for the relevance score"
    )


class QueryAnalysis(BaseModel):
    """Analysis of the user's query."""
    intent: str = Field(description="The main intent of the query")
    needs_web_search: bool = Field(description="Whether web search is needed")
    is_answerable: bool = Field(description="Whether the query can be answered")
    gaps: List[str] = Field(description="Information gaps that need to be filled")


class AgentState(TypedDict):
    """State schema for the Agentic RAG graph."""
    
    # User Input
    question: str
    
    # Retrieval State
    documents: List[Document]
    original_documents: List[Document]
    retrieval_count: int
    
    # Web Search State
    web_search_needed: bool
    web_search_results: List[dict]
    
    # Grading State
    document_scores: List[GradeScore]
    relevant_documents: List[Document]
    hallucination_score: float
    
    # Generation State
    generation: str
    generation_attempts: int
    
    # Routing State
    route: str
    reasoning: str
    
    # Metadata
    iteration: int
    max_iterations_reached: bool
    query_analysis: Optional[QueryAnalysis]
    
    # Observability
    context_precision: float
    sources_used: List[str]


class RouteResponse(BaseModel):
    """Response model for routing decisions."""
    route: Literal["vectorstore", "websearch", "generate"]
    reasoning: str


def determine_route(state: AgentState) -> str:
    """
    Determine the next route based on current state.
    
    Routes:
    - "retrieve": Fetch context from vector DB
    - "websearch": Use DuckDuckGo/Tavily for web search
    - "grade": Evaluate retrieved documents
    - "generate": Synthesize final answer
    - "end": End the conversation
    """
    if state.get("iteration", 0) >= state.get("max_retrieval_iterations", 3):
        return "generate"
    
    if not state.get("documents") and not state.get("web_search_results"):
        return "retrieve"
    
    if state.get("web_search_needed") and not state.get("web_search_results"):
        return "websearch"
    
    if state.get("relevant_documents"):
        return "generate"
    
    return "grade"


def should_continue(state: AgentState) -> str:
    """Determine if the graph should continue or end."""
    if state.get("max_iterations_reached"):
        return "end"
    
    if state.get("generation"):
        return "end"
    
    return "continue"


class GraphConfig:
    """Configuration for the LangGraph workflow."""
    
    def __init__(
        self,
        max_retrieval_iterations: int = 3,
        min_relevance_score: float = 0.5,
        hallucination_threshold: float = 0.7,
        enable_web_search: bool = True
    ):
        self.max_retrieval_iterations = max_retrieval_iterations
        self.min_relevance_score = min_relevance_score
        self.hallucination_threshold = hallucination_threshold
        self.enable_web_search = enable_web_search


def create_initial_state(question: str) -> AgentState:
    """
    Create initial state for a new query.
    
    Args:
        question: The user's question (must be a string)
        
    Returns:
        AgentState with all required fields initialized
    """
    # Ensure question is a valid string
    if not isinstance(question, str):
        raise ValueError(f"question must be a string, got {type(question)}: {question}")
    
    # Clean and validate the question
    question = question.strip()
    if not question:
        raise ValueError("question cannot be empty")
    
    return AgentState(
        question=question,
        documents=[],
        original_documents=[],
        retrieval_count=0,
        web_search_needed=False,
        web_search_results=[],
        document_scores=[],
        relevant_documents=[],
        hallucination_score=0.0,
        generation="",
        generation_attempts=0,
        route="retrieve",  # Always start with retrieve
        reasoning="Initial state created",
        iteration=0,
        max_iterations_reached=False,
        query_analysis=None,
        context_precision=0.0,
        sources_used=[]
    )


def validate_state(state: AgentState) -> bool:
    """
    Validate that the state has the correct structure and types.
    
    Args:
        state: The state to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Check required string fields
    if not isinstance(state.get("question"), str):
        raise ValueError(f"question must be a string, got {type(state.get('question'))}")
    
    if not isinstance(state.get("route"), str):
        raise ValueError(f"route must be a string, got {type(state.get('route'))}")
    
    if not isinstance(state.get("reasoning"), str):
        raise ValueError(f"reasoning must be a string, got {type(state.get('reasoning'))}")
    
    if not isinstance(state.get("generation"), str):
        raise ValueError(f"generation must be a string, got {type(state.get('generation'))}")
    
    # Check required numeric fields
    if not isinstance(state.get("iteration"), int):
        raise ValueError(f"iteration must be an int, got {type(state.get('iteration'))}")
    
    if not isinstance(state.get("retrieval_count"), int):
        raise ValueError(f"retrieval_count must be an int, got {type(state.get('retrieval_count'))}")
    
    if not isinstance(state.get("generation_attempts"), int):
        raise ValueError(f"generation_attempts must be an int, got {type(state.get('generation_attempts'))}")
    
    # Check boolean fields
    if not isinstance(state.get("web_search_needed"), bool):
        raise ValueError(f"web_search_needed must be a bool, got {type(state.get('web_search_needed'))}")
    
    if not isinstance(state.get("max_iterations_reached"), bool):
        raise ValueError(f"max_iterations_reached must be a bool, got {type(state.get('max_iterations_reached'))}")
    
    # Check list fields
    if not isinstance(state.get("documents"), list):
        raise ValueError(f"documents must be a list, got {type(state.get('documents'))}")
    
    if not isinstance(state.get("relevant_documents"), list):
        raise ValueError(f"relevant_documents must be a list, got {type(state.get('relevant_documents'))}")
    
    if not isinstance(state.get("web_search_results"), list):
        raise ValueError(f"web_search_results must be a list, got {type(state.get('web_search_results'))}")
    
    if not isinstance(state.get("sources_used"), list):
        raise ValueError(f"sources_used must be a list, got {type(state.get('sources_used'))}")
    
    return True

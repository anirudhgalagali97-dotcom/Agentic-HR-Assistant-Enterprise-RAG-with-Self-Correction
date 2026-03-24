"""
LangGraph Workflow for Agentic RAG System
Implements the stateful graph with conditional routing
"""
from typing import Dict, Any, Literal, Callable
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from .state import AgentState, create_initial_state, GraphConfig
from .nodes import (
    analyze_query_node,
    retrieve_documents_node,
    grade_documents_node,
    websearch_node,
    generate_answer_node,
    decide_route_node,
    check_hallucination_node
)


logger = logging.getLogger(__name__)


def create_agentic_rag_graph(config: GraphConfig = None) -> StateGraph:
    """
    Create the Agentic RAG LangGraph workflow.
    
    The graph implements the following flow:
    1. Analyze Query - Understand user intent
    2. Retrieve - Fetch from vector store
    3. Grade Documents - Evaluate relevance (Hallucination Guardrail)
    4. Web Search (optional) - Fetch from web if local docs insufficient
    5. Generate Answer - Synthesize response
    6. Check Hallucination - Final quality check
    """
    config = config or GraphConfig()
    
    # Define the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_query_node)
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("websearch", websearch_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("decide", decide_route_node)
    workflow.add_node("check_hallucination", check_hallucination_node)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Add conditional edges based on route
    def route_after_analyze(state: AgentState) -> str:
        """Route after query analysis."""
        return state.get("route", "retrieve")
    
    def route_after_retrieve(state: AgentState) -> str:
        """Route after retrieval."""
        return state.get("route", "grade")
    
    def route_after_grade(state: AgentState) -> str:
        """Route after grading."""
        route = state.get("route", "generate")
        return route
    
    def route_after_websearch(state: AgentState) -> str:
        """Route after web search."""
        return state.get("route", "generate")
    
    def route_after_decide(state: AgentState) -> str:
        """Route after decision."""
        return state.get("route", "retrieve")
    
    def route_after_generate(state: AgentState) -> str:
        """Route after generation."""
        # Always check for hallucinations
        if state.get("relevant_documents"):
            return "check_hallucination"
        return END
    
    def route_after_check(state: AgentState) -> str:
        """Route after hallucination check."""
        if state.get("route") == "generate":
            return "generate"
        return END
    
    # Add edges with routing
    workflow.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "retrieve": "retrieve",
            "websearch": "websearch",
            "generate": "generate",
            "end": END
        }
    )
    
    workflow.add_edge("retrieve", "grade")
    
    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "retrieve": "retrieve",
            "websearch": "websearch",
            "generate": "generate"
        }
    )
    
    workflow.add_conditional_edges(
        "websearch",
        route_after_websearch,
        {
            "generate": "generate",
            "retrieve": "retrieve"
        }
    )
    
    workflow.add_edge("generate", "check_hallucination")
    
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_check,
        {
            "generate": "generate",
            "end": END
        }
    )
    
    # Compile the graph
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Agentic RAG graph compiled successfully")
    
    return compiled_graph


class AgenticRAGAgent:
    """Wrapper class for the Agentic RAG agent."""
    
    def __init__(self, config: GraphConfig = None):
        self.config = config or GraphConfig()
        self.graph = create_agentic_rag_graph(self.config)
        self._initialize()
    
    def _initialize(self):
        """Initialize the agent."""
        logger.info("Initializing Agentic RAG Agent")
    
    def invoke(self, question: str, thread_id: str = None) -> Dict[str, Any]:
        """
        Invoke the agent with a question.
        
        Args:
            question: The user's question
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Dictionary containing the answer and metadata
        """
        logger.info(f"Invoking agent with question: {question[:100]}...")
        
        # Create initial state
        initial_state = create_initial_state(question)
        
        # Configure thread for memory
        config = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        
        # Run the graph
        try:
            result = self.graph.invoke(initial_state, config=config)
            
            return {
                "answer": result.get("generation", ""),
                "question": question,
                "sources": result.get("sources_used", []),
                "context_precision": result.get("context_precision", 0.0),
                "hallucination_score": result.get("hallucination_score", 0.0),
                "iterations": result.get("iteration", 0),
                "route": result.get("route", "unknown"),
                "reasoning": result.get("reasoning", ""),
                "relevant_documents": result.get("relevant_documents", []),
                "web_search_results": result.get("web_search_results", []),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            return {
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "question": question,
                "status": "error",
                "error": str(e)
            }
    
    async def ainvoke(self, question: str, thread_id: str = None) -> Dict[str, Any]:
        """Async version of invoke."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.invoke, question, thread_id
        )
    
    def get_graph_diagram(self) -> str:
        """Get the graph structure as ASCII diagram."""
        return """
Agentic RAG Workflow
====================

┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   ANALYZE   │ ──► Determine query intent
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  RETRIEVE   │ ──► Fetch from vector store
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    GRADE    │ ──► Evaluate document relevance
└──────┬──────┘
       │
       ├──────────┐
       │          │
       ▼          ▼
┌──────────┐  ┌────────────┐
│  REPEAT  │  │  WEBSearch │ ──► (if no relevant docs)
└──────────┘  └──────┬─────┘
                    │
                    ▼
             ┌─────────────┐
             │  GENERATE   │ ──► Synthesize answer
             └──────┬──────┘
                    │
                    ▼
             ┌──────────────┐
             │CHECK HALLUCI │ ──► Final quality check
             └──────┬───────┘
                    │
                    ▼
             ┌─────────────┐
             │    END      │
             └─────────────┘
        """


# Global agent instance
_agent_instance = None


def get_agent(config: GraphConfig = None) -> AgenticRAGAgent:
    """Get or create the global agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AgenticRAGAgent(config)
    return _agent_instance


def reset_agent():
    """Reset the global agent instance."""
    global _agent_instance
    _agent_instance = None

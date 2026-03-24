"""
LangGraph Nodes for Agentic RAG System
Implements all the nodes: Retrieve, WebSearch, Grade, Generate
"""
from typing import List, Dict, Any, Optional
import logging
import json
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS

from config.settings import settings, get_openai_api_key
from .state import AgentState, GradeScore, QueryAnalysis


logger = logging.getLogger(__name__)


# LLM Initialization
def get_llm(model: str = None, temperature: float = 0.0) -> ChatOpenAI:
    """Get configured LLM instance."""
    return ChatOpenAI(
        model=model or settings.llm_model,
        temperature=temperature,
        api_key=get_openai_api_key()
    )


# ============================================================
# NODE 1: Analyze Query
# ============================================================
def analyze_query_node(state: AgentState) -> Dict[str, Any]:
    """Analyze the user query to determine intent and search strategy."""
    logger.info(f"Analyzing query: {state['question']}")
    
    llm = get_llm(temperature=0.3)
    
    prompt = ChatPromptTemplate.from_template(
        """Analyze the following query and determine:
        1. The main intent and topic
        2. Whether this query requires information from the local document database
        3. Whether this query requires real-time web search
        4. Whether the query can be answered from general knowledge
        
        Question: {question}
        
        Return a JSON object with:
        - intent: str (main topic/intent)
        - needs_document_search: bool
        - needs_web_search: bool
        - is_answerable: bool
        - gaps: list of strings (information gaps)"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"question": state["question"]})
        analysis = json.loads(result)
        
        return {
            "query_analysis": QueryAnalysis(
                intent=analysis.get("intent", "general"),
                needs_web_search=analysis.get("needs_web_search", False),
                is_answerable=analysis.get("is_answerable", True),
                gaps=analysis.get("gaps", [])
            ),
            "web_search_needed": analysis.get("needs_web_search", False) or analysis.get("needs_document_search", True),
            "route": "retrieve",
            "reasoning": f"Query analysis complete. Intent: {analysis.get('intent')}"
        }
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return {
            "query_analysis": QueryAnalysis(
                intent="general",
                needs_web_search=True,
                is_answerable=False,
                gaps=["Unable to analyze query"]
            ),
            "web_search_needed": True,
            "route": "retrieve",
            "reasoning": "Default routing due to analysis failure"
        }


# ============================================================
# NODE 2: Retrieve Documents
# ============================================================
def retrieve_documents_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve documents from the vector store using hybrid retrieval."""
    from retrieval.retriever import create_hybrid_retriever
    from retrieval.self_query import create_self_query_retriever
    
    logger.info(f"Retrieving documents for: {state['question']}")
    
    try:
        # Create hybrid retriever
        hybrid_retriever = create_hybrid_retriever(
            vector_weight=settings.ensemble_weights[1],
            bm25_weight=settings.ensemble_weights[0]
        )
        
        # Also create self-querying retriever
        llm = get_llm(temperature=0)
        self_query_retriever = create_self_query_retriever(
            vectorstore=hybrid_retriever.vectorstore,
            llm=llm
        )
        
        # Try self-querying retrieval first
        try:
            documents = self_query_retriever.invoke(state["question"])
        except Exception as e:
            logger.warning(f"Self-query retrieval failed: {e}, falling back to hybrid")
            documents = hybrid_retriever.invoke(state["question"])
        
        retrieval_count = state.get("retrieval_count", 0) + 1
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        return {
            "documents": documents,
            "original_documents": documents.copy() if retrieval_count == 1 else state.get("original_documents", []),
            "retrieval_count": retrieval_count,
            "iteration": state.get("iteration", 0) + 1,
            "route": "grade",
            "reasoning": f"Retrieved {len(documents)} documents from vector store"
        }
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "documents": [],
            "route": "websearch",
            "reasoning": f"Retrieval failed: {str(e)}, switching to web search"
        }


# ============================================================
# NODE 3: Grade Documents (Hallucination Guardrail)
# ============================================================
def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """Grade retrieved documents for relevance to the question."""
    logger.info("Grading documents for relevance")
    
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        logger.warning("No documents to grade")
        return {
            "document_scores": [],
            "relevant_documents": [],
            "hallucination_score": 1.0,
            "context_precision": 0.0,
            "route": "websearch",
            "reasoning": "No documents retrieved for grading"
        }
    
    llm = get_llm(temperature=0)
    
    # Grade prompt
    grade_prompt = ChatPromptTemplate.from_template(
        """You are a relevance grader. Your task is to determine if a retrieved document 
        is relevant to the user's question.
        
        Consider:
        1. Does the document contain information that helps answer the question?
        2. Is the document topically related to the question?
        3. Does the document provide factual context for the question?
        
        Retrieved Document:
        {document}
        
        User Question: {question}
        
        Return a JSON object with:
        - binary_score: "yes" or "no"
        - score: float between 0 and 1
        - reasoning: brief explanation of why the document is or isn't relevant"""
    )
    
    graded_scores: List[GradeScore] = []
    relevant_docs: List[Document] = []
    total_score = 0.0
    
    for doc in documents:
        try:
            chain = grade_prompt | llm | StrOutputParser()
            result = chain.invoke({
                "document": doc.page_content[:1000],  # Limit context
                "question": question
            })
            
            grade_result = json.loads(result)
            score = GradeScore(
                binary_score=grade_result.get("binary_score", "no"),
                score=grade_result.get("score", 0.0),
                reasoning=grade_result.get("reasoning", "")
            )
            
            graded_scores.append(score)
            total_score += score.score
            
            if score.binary_score.lower() == "yes" or score.score >= settings.min_relevance_score:
                relevant_docs.append(doc)
                
        except Exception as e:
            logger.error(f"Grading failed for document: {e}")
            graded_scores.append(GradeScore(
                binary_score="no",
                score=0.0,
                reasoning=f"Grading error: {str(e)}"
            ))
    
    # Calculate metrics
    avg_score = total_score / len(documents) if documents else 0.0
    context_precision = len(relevant_docs) / len(documents) if documents else 0.0
    hallucination_score = 1.0 - avg_score
    
    # Determine routing
    if relevant_docs:
        route = "generate"
        reasoning = f"Found {len(relevant_docs)} relevant documents (precision: {context_precision:.2f})"
    elif state.get("iteration", 0) < settings.max_retrieval_iterations:
        route = "retrieve"
        reasoning = "No relevant documents found, re-attempting retrieval"
    else:
        route = "websearch"
        reasoning = "No relevant documents after max iterations, attempting web search"
    
    logger.info(f"Grading complete: {len(relevant_docs)}/{len(documents)} relevant, "
                f"precision: {context_precision:.2f}")
    
    return {
        "document_scores": graded_scores,
        "relevant_documents": relevant_docs,
        "hallucination_score": hallucination_score,
        "context_precision": context_precision,
        "route": route,
        "reasoning": reasoning,
        "sources_used": [doc.metadata.get("source", "unknown") for doc in relevant_docs]
    }


# ============================================================
# NODE 4: Web Search
# ============================================================
def websearch_node(state: AgentState) -> Dict[str, Any]:
    """Perform web search using DuckDuckGo."""
    logger.info(f"Performing web search for: {state['question']}")
    
    question = state["question"]
    max_results = settings.max_web_search_results
    
    try:
        # Use DuckDuckGo for web search
        search_results = []
        
        with DDGS() as ddgs:
            results = ddgs.text(question, max_results=max_results)
            for r in results:
                search_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": "duckduckgo"
                })
        
        logger.info(f"Web search returned {len(search_results)} results")
        
        return {
            "web_search_results": search_results,
            "web_search_needed": False,
            "route": "generate",
            "reasoning": f"Web search found {len(search_results)} results"
        }
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "web_search_results": [],
            "route": "generate",
            "reasoning": f"Web search failed: {str(e)}, generating answer with available context"
        }


# ============================================================
# NODE 5: Generate Answer
# ============================================================
def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """Generate the final answer using retrieved context."""
    logger.info("Generating answer")
    
    question = state["question"]
    relevant_docs = state.get("relevant_documents", [])
    web_results = state.get("web_search_results", [])
    
    # Prepare context
    context_parts = []
    
    # Add document context
    if relevant_docs:
        context_parts.append("=== DOCUMENT CONTEXT ===")
        for i, doc in enumerate(relevant_docs[:5], 1):  # Limit to top 5
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"\n[Document {i}] (Source: {source})\n{doc.page_content}")
    
    # Add web search context
    if web_results:
        context_parts.append("\n=== WEB SEARCH RESULTS ===")
        for i, result in enumerate(web_results[:3], 1):
            context_parts.append(f"\n[Result {i}] {result.get('title', 'No title')}")
            context_parts.append(f"URL: {result.get('url', 'No URL')}")
            context_parts.append(f"Content: {result.get('snippet', 'No content')}")
    
    context = "\n".join(context_parts) if context_parts else "No relevant context found."
    
    # Generation prompt
    generation_prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant that answers questions based on the provided context.
        
        Instructions:
        1. Use ONLY the provided context to answer the question
        2. If the context doesn't contain enough information, say so clearly
        3. Cite your sources when using information from documents
        4. Be concise but thorough
        5. If you're uncertain about something, indicate that
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
    )
    
    llm = get_llm(temperature=0.3)
    chain = generation_prompt | llm | StrOutputParser()
    
    try:
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        generation_attempts = state.get("generation_attempts", 0) + 1
        
        logger.info(f"Generated answer (attempt {generation_attempts})")
        
        return {
            "generation": answer,
            "generation_attempts": generation_attempts,
            "route": "end",
            "reasoning": "Answer generated successfully"
        }
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            "generation": f"I apologize, but I encountered an error generating the answer: {str(e)}",
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "route": "end",
            "reasoning": f"Generation failed: {str(e)}"
        }


# ============================================================
# NODE 6: Decide Route
# ============================================================
def decide_route_node(state: AgentState) -> Dict[str, Any]:
    """Decide the next route based on current state."""
    logger.info("Deciding next route")
    
    iteration = state.get("iteration", 0)
    max_iterations = settings.max_retrieval_iterations
    
    # Check if max iterations reached
    if iteration >= max_iterations:
        return {
            "route": "generate",
            "max_iterations_reached": True,
            "reasoning": f"Max iterations ({max_iterations}) reached"
        }
    
    # Check if we have relevant documents
    if state.get("relevant_documents"):
        return {
            "route": "generate",
            "reasoning": "Relevant documents found, proceeding to generate"
        }
    
    # Check if web search is needed
    if state.get("web_search_needed") and not state.get("web_search_results"):
        return {
            "route": "websearch",
            "reasoning": "Web search needed but not yet performed"
        }
    
    # Default to retrieval
    return {
        "route": "retrieve",
        "reasoning": "Continuing retrieval process"
    }


# ============================================================
# NODE 7: Check Hallucination
# ============================================================
def check_hallucination_node(state: AgentState) -> Dict[str, Any]:
    """Check if the generated answer might contain hallucinations."""
    logger.info("Checking for hallucinations")
    
    question = state["question"]
    answer = state.get("generation", "")
    relevant_docs = state.get("relevant_documents", [])
    
    if not answer or not relevant_docs:
        return {
            "route": "end",
            "reasoning": "No answer or context to check"
        }
    
    llm = get_llm(temperature=0)
    
    # Hallucination check prompt
    check_prompt = ChatPromptTemplate.from_template(
        """You are a factual accuracy checker. Evaluate if the answer is supported by the context.
        
        Question: {question}
        
        Answer: {answer}
        
        Context: {context}
        
        Return a JSON object with:
        - is_factual: bool (is the answer supported by context?)
        - confidence: float 0-1 (confidence in factual accuracy)
        - issues: list of strings (any potential hallucinations or inaccuracies)
        - suggestions: list of strings (how to improve the answer)"""
    )
    
    context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
    
    try:
        chain = check_prompt | llm | StrOutputParser()
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context
        })
        
        check_result = json.loads(result)
        
        if not check_result.get("is_factual", True):
            logger.warning(f"Hallucination detected: {check_result.get('issues', [])}")
        
        return {
            "hallucination_score": 1.0 - check_result.get("confidence", 1.0),
            "route": "end" if check_result.get("is_factual", True) else "generate",
            "reasoning": f"Hallucination check complete. Confidence: {check_result.get('confidence', 0):.2f}"
        }
    except Exception as e:
        logger.error(f"Hallucination check failed: {e}")
        return {
            "route": "end",
            "reasoning": "Hallucination check failed, returning answer"
        }


# ============================================================
# NODE MAPPING
# ============================================================
NODE_MAPPING = {
    "analyze": analyze_query_node,
    "retrieve": retrieve_documents_node,
    "grade": grade_documents_node,
    "websearch": websearch_node,
    "generate": generate_answer_node,
    "decide": decide_route_node,
    "check_hallucination": check_hallucination_node
}


def get_node(node_name: str):
    """Get a node by name."""
    return NODE_MAPPING.get(node_name)

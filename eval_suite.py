"""
Evaluation Suite for Agentic RAG System
=======================================
Uses DeepEval library to evaluate the RAG pipeline with:
- FaithfulnessMetric: Measures hallucination/inaccuracy
- AnswerRelevancyMetric: Measures how relevant the answer is to the question
- ContextualPrecisionMetric: Measures retrieval quality

Run: python eval_suite.py
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DeepEval imports
try:
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval import evaluate
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("DeepEval not installed. Run: pip install deepeval")

# Import the agent graph
from agents.graph import get_agent, AgenticRAGAgent
from config.settings import get_ollama_config, settings


# HR-Related Test Questions
HR_TEST_QUESTIONS = [
    "What is the company's policy on remote work and work-from-home arrangements?",
    "Can you explain the employee benefits package including health insurance and retirement plans?",
    "What is the procedure for requesting vacation leave and how many days are provided annually?",
    "What are the company's guidelines for professional development and training opportunities?",
    "How does the performance review process work and what criteria are used for evaluation?"
]


@dataclass
class EvaluationResult:
    """Stores results for a single test case."""
    question: str
    answer: str
    expected_output: Optional[str] = None
    retrieval_context: List[str] = field(default_factory=list)
    
    # Metrics
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    contextual_precision_score: Optional[float] = None
    
    # Status
    status: str = "pending"
    error: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary statistics for the evaluation suite."""
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_contextual_precision: float = 0.0
    
    results: List[EvaluationResult] = field(default_factory=list)


class AgenticRAGEvaluator:
    """
    Evaluation framework for the Agentic RAG system.
    
    Uses DeepEval metrics to evaluate:
    - Faithfulness: Whether the answer is faithful to the context
    - Answer Relevancy: Whether the answer is relevant to the question
    - Contextual Precision: Quality of retrieval
    """
    
    def __init__(
        self,
        agent: Optional[AgenticRAGAgent] = None,
        use_cache: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            agent: Optional pre-initialized agent (will create if not provided)
            use_cache: Whether to use cached results if available
        """
        self.agent = agent
        self.use_cache = use_cache
        self._cache: Dict[str, EvaluationResult] = {}
        
        if not DEEPEVAL_AVAILABLE:
            logger.error("DeepEval is required but not installed")
            raise ImportError("Please install deepeval: pip install deepeval")
    
    def _get_agent(self) -> AgenticRAGAgent:
        """Get or create the agent instance."""
        if self.agent is None:
            logger.info("Initializing agent for evaluation...")
            self.agent = get_agent()
        return self.agent
    
    def _create_test_case(
        self,
        question: str,
        answer: str,
        context: List[str]
    ) -> LLMTestCase:
        """Create a DeepEval test case."""
        return LLMTestCase(
            input=question,
            actual_output=answer,
            context=context,
            retrieval_context=context
        )
    
    def _run_metric(
        self,
        metric,
        test_case: LLMTestCase,
        metric_name: str
    ) -> float:
        """Run a single metric on a test case."""
        try:
            metric.measure(test_case)
            score = metric.score
            logger.info(f"{metric_name}: {score:.4f}")
            return score
        except Exception as e:
            logger.error(f"Error running {metric_name}: {e}")
            return 0.0
    
    def evaluate_single(
        self,
        question: str,
        expected_output: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single question.
        
        Args:
            question: The question to evaluate
            expected_output: Optional expected answer for comparison
            
        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult(
            question=question,
            answer="",  # Initialize with empty answer, will be updated after agent call
            expected_output=expected_output,
            status="running"
        )
        
        # Check cache
        if self.use_cache and question in self._cache:
            logger.info(f"Using cached result for: {question[:50]}...")
            return self._cache[question]
        
        try:
            # Get agent and run query
            agent = self._get_agent()
            logger.info(f"Running query: {question[:50]}...")
            
            agent_result = agent.invoke(question)
            
            # Extract the generated answer from the agent response
            generated_answer = agent_result.get("answer", "No answer generated")
            
            # Create result with the generated answer
            result = EvaluationResult(
                question=question,
                answer=generated_answer,
                expected_output=expected_output,
                status="completed" if agent_result.get("status") == "success" else "failed"
            )
            
            # Extract retrieval context
            context = []
            for doc in agent_result.get("relevant_documents", []):
                context.append(doc.page_content)
            
            # Add web search results if available
            for web_result in agent_result.get("web_search_results", []):
                context.append(web_result.get("snippet", ""))
            
            result.retrieval_context = context
            
            if not context:
                result.error = "No retrieval context available"
                result.status = "failed"
                return result
            
            # Create test case
            test_case = self._create_test_case(question, result.answer, context)
            
            # Initialize metrics
            faithfulness_metric = FaithfulnessMetric(threshold=0.7)
            answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
            contextual_precision_metric = ContextualPrecisionMetric(threshold=0.7)
            
            # Run metrics
            logger.info("Calculating FaithfulnessMetric...")
            result.faithfulness_score = self._run_metric(
                faithfulness_metric, test_case, "Faithfulness"
            )
            
            logger.info("Calculating AnswerRelevancyMetric...")
            result.answer_relevancy_score = self._run_metric(
                answer_relevancy_metric, test_case, "Answer Relevancy"
            )
            
            logger.info("Calculating ContextualPrecisionMetric...")
            result.contextual_precision_score = self._run_metric(
                contextual_precision_metric, test_case, "Contextual Precision"
            )
            
            # Cache result
            if self.use_cache:
                self._cache[question] = result
            
            logger.info(f"Evaluation complete for: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            result.status = "failed"
            result.error = str(e)
        
        return result
    
    def evaluate_batch(
        self,
        questions: List[str],
        expected_outputs: Optional[List[str]] = None
    ) -> EvaluationSummary:
        """
        Evaluate multiple questions.
        
        Args:
            questions: List of questions to evaluate
            expected_outputs: Optional list of expected answers
            
        Returns:
            EvaluationSummary with aggregated results
        """
        summary = EvaluationSummary(total_tests=len(questions))
        expected_outputs = expected_outputs or [None] * len(questions)
        
        logger.info(f"Starting batch evaluation of {len(questions)} questions...")
        
        for i, (question, expected) in enumerate(zip(questions, expected_outputs), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i}/{len(questions)}")
            logger.info(f"{'='*60}")
            
            result = self.evaluate_single(question, expected)
            summary.results.append(result)
            
            if result.status == "completed":
                summary.successful_tests += 1
            else:
                summary.failed_tests += 1
        
        # Calculate averages
        summary = self._calculate_summary(summary)
        
        return summary
    
    def _calculate_summary(self, summary: EvaluationSummary) -> EvaluationSummary:
        """Calculate summary statistics."""
        completed = [r for r in summary.results if r.status == "completed"]
        
        if completed:
            summary.avg_faithfulness = sum(
                r.faithfulness_score for r in completed
            ) / len(completed)
            
            summary.avg_answer_relevancy = sum(
                r.answer_relevancy_score for r in completed
            ) / len(completed)
            
            summary.avg_contextual_precision = sum(
                r.contextual_precision_score for r in completed
            ) / len(completed)
        
        return summary
    
    def print_results_table(self, summary: EvaluationSummary):
        """Print results in a formatted table."""
        print("\n" + "=" * 100)
        print("EVALUATION RESULTS - AGENTIC RAG SYSTEM")
        print("=" * 100)
        
        # Individual test results
        headers = ["#", "Question", "Faithfulness", "Relevancy", "Precision", "Status"]
        table_data = []
        
        for i, result in enumerate(summary.results, 1):
            question_short = result.question[:40] + "..." if len(result.question) > 40 else result.question
            
            table_data.append([
                i,
                question_short,
                f"{result.faithfulness_score:.4f}" if result.faithfulness_score else "N/A",
                f"{result.answer_relevancy_score:.4f}" if result.answer_relevancy_score else "N/A",
                f"{result.contextual_precision_score:.4f}" if result.contextual_precision_score else "N/A",
                result.status.upper()
            ])
        
        print("\nIndividual Test Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Summary table
        print("\n" + "-" * 100)
        print("SUMMARY STATISTICS")
        print("-" * 100)
        
        summary_data = [
            ["Total Tests", summary.total_tests],
            ["Successful", summary.successful_tests],
            ["Failed", summary.failed_tests],
            ["Success Rate", f"{(summary.successful_tests/summary.total_tests)*100:.1f}%"],
            ["", ""],
            ["Average Faithfulness", f"{summary.avg_faithfulness:.4f}"],
            ["Average Answer Relevancy", f"{summary.avg_answer_relevancy:.4f}"],
            ["Average Contextual Precision", f"{summary.avg_contextual_precision:.4f}"],
            ["", ""],
            ["Overall Score (avg)", f"{(summary.avg_faithfulness + summary.avg_answer_relevancy + summary.avg_contextual_precision)/3:.4f}"]
        ]
        
        print(tabulate(summary_data, tablefmt="grid"))
        
        # Pass/Fail status
        print("\n" + "-" * 100)
        threshold = 0.7
        print(f"PIPELINE STATUS (threshold = {threshold})")
        print("-" * 100)
        
        metrics_status = [
            ["Faithfulness", "PASS" if summary.avg_faithfulness >= threshold else "FAIL"],
            ["Answer Relevancy", "PASS" if summary.avg_answer_relevancy >= threshold else "FAIL"],
            ["Contextual Precision", "PASS" if summary.avg_contextual_precision >= threshold else "FAIL"]
        ]
        
        print(tabulate(metrics_status, headers=["Metric", "Status"], tablefmt="grid"))
        
        overall_pass = (
            summary.avg_faithfulness >= threshold and
            summary.avg_answer_relevancy >= threshold and
            summary.avg_contextual_precision >= threshold
        )
        
        print(f"\n{'='*100}")
        print(f"OVERALL PIPELINE: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        print(f"{'='*100}\n")
    
    def export_results(
        self,
        summary: EvaluationSummary,
        output_path: str = "evaluation_results.json"
    ):
        """Export results to JSON file."""
        import json
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": summary.total_tests,
                "successful_tests": summary.successful_tests,
                "failed_tests": summary.failed_tests,
                "avg_faithfulness": summary.avg_faithfulness,
                "avg_answer_relevancy": summary.avg_answer_relevancy,
                "avg_contextual_precision": summary.avg_contextual_precision
            },
            "results": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "faithfulness_score": r.faithfulness_score,
                    "answer_relevancy_score": r.answer_relevancy_score,
                    "contextual_precision_score": r.contextual_precision_score,
                    "status": r.status,
                    "error": r.error
                }
                for r in summary.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to {output_path}")


def run_evaluation_suite():
    """
    Main function to run the evaluation suite.
    
    Runs 5 HR-related questions through the Agentic RAG system
    and evaluates using DeepEval metrics.
    """
    print("\n" + "=" * 100)
    print("AGENTIC RAG EVALUATION SUITE")
    print("Using DeepEval Metrics")
    print("=" * 100)
    
    if not DEEPEVAL_AVAILABLE:
        print("\n❌ ERROR: DeepEval is not installed.")
        print("Please install it with: pip install deepeval")
        print("\nNote: DeepEval requires an OpenAI API key for evaluation.")
        return None
    
    # Verify Ollama configuration
    try:
        base_url, model = get_ollama_config()
        print(f"\n[INFO] Using Ollama: {model} at {base_url}")
    except Exception as e:
        print(f"\n⚠️ WARNING: Ollama configuration error: {e}")
    
    # Initialize evaluator
    try:
        evaluator = AgenticRAGEvaluator()
    except ImportError as e:
        print(f"\n❌ Initialization failed: {e}")
        return None
    
    # Run evaluation on HR questions
    print(f"\n📋 Running evaluation on {len(HR_TEST_QUESTIONS)} HR-related questions...\n")
    
    start_time = datetime.now()
    summary = evaluator.evaluate_batch(HR_TEST_QUESTIONS)
    end_time = datetime.now()
    
    # Print results
    evaluator.print_results_table(summary)
    
    # Export results
    evaluator.export_results(summary, "evaluation_results.json")
    
    # Print timing
    duration = (end_time - start_time).total_seconds()
    print(f"Evaluation completed in {duration:.2f} seconds")
    
    return summary


if __name__ == "__main__":
    run_evaluation_suite()

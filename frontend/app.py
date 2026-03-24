"""
Gradio Frontend for Agentic RAG System
Provides a user-friendly web interface for querying documents
"""
import gradio as gr
from gradio.components import Textbox, TextArea, Button, Dropdown, Slider
import requests
import json
import logging
import os
from typing import Optional, List, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class AgenticRAGUI:
    """Gradio UI for Agentic RAG System."""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url
        self.history: List[Dict[str, str]] = []
    
    def check_api_health(self) -> tuple[bool, str]:
        """Check if the API is running."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return True, f"API Status: {data.get('status', 'unknown')}"
            return False, f"API returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "API not running. Please start the FastAPI server."
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_document_stats(self) -> str:
        """Get document statistics."""
        try:
            response = requests.get(f"{self.api_base_url}/stats/documents", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return f"""Documents: {data.get('document_count', 0)}
Collection: {data.get('collection_name', 'N/A')}
Embedding: {data.get('embedding_model', 'N/A')}
Chunk Size: {data.get('chunk_size', 0)}"""
            return "Unable to fetch document stats"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_system_stats(self) -> str:
        """Get system statistics."""
        try:
            response = requests.get(f"{self.api_base_url}/stats/system", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return f"""Total Queries: {data.get('total_queries', 0)}
Total Errors: {data.get('total_errors', 0)}
Error Rate: {data.get('error_rate', 0):.2%}
Avg Latency: {data.get('avg_latency_ms', 0):.0f}ms
Avg Precision: {data.get('avg_context_precision', 0):.2%}"""
            return "Unable to fetch system stats"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def query(
        self,
        question: str,
        show_sources: bool = True,
        progress=gr.Progress()
    ) -> tuple[str, str, str]:
        """
        Query the RAG system.
        
        Returns:
            Tuple of (answer, sources_display, stats_display)
        """
        if not question.strip():
            return "Please enter a question.", "", ""
        
        progress(0.1, desc="Analyzing query...")
        self.history.append({"role": "user", "content": question})
        
        try:
            progress(0.2, desc="Processing request...")
            response = requests.post(
                f"{self.api_base_url}/query",
                json={
                    "question": question,
                    "include_sources": show_sources
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                
                progress(0.9, desc="Formatting response...")
                
                answer = data.get("answer", "No answer generated.")
                
                # Add to history
                self.history.append({"role": "assistant", "content": answer})
                
                # Format sources
                sources_text = ""
                if show_sources:
                    sources = data.get("sources", [])
                    if sources:
                        sources_text = "### Sources\n\n"
                        for i, source in enumerate(sources, 1):
                            if source.get("source") == "web":
                                sources_text += f"**{i}. Web:** [{source.get('title', 'No title')}]({source.get('url', '#')})\n"
                                sources_text += f"   {source.get('snippet', 'No snippet')[:200]}...\n\n"
                            else:
                                sources_text += f"**{i}. Document:** {source.get('file_name', 'Unknown')}\n"
                                sources_text += f"   {source.get('content', '')[:300]}...\n\n"
                
                # Format stats
                stats_text = f"""**Query Stats:**
- Latency: {data.get('latency_ms', 0):.0f}ms
- Context Precision: {data.get('context_precision', 0):.2%}
- Hallucination Score: {data.get('hallucination_score', 0):.2%}
- Iterations: {data.get('iterations', 0)}
- Web Search Used: {'Yes' if data.get('web_search_used') else 'No'}"""
                
                progress(1.0, desc="Complete!")
                return answer, sources_text, stats_text
                
            else:
                error_msg = f"Error: API returned status {response.status_code}"
                logger.error(error_msg)
                return error_msg, "", ""
                
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again.", "", ""
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Error: {str(e)}", "", ""
    
    def ingest_documents(self, progress=gr.Progress()) -> str:
        """Trigger document ingestion."""
        progress(0.1, desc="Starting ingestion...")
        try:
            response = requests.post(f"{self.api_base_url}/ingest", timeout=5)
            if response.status_code == 200:
                progress(1.0, desc="Ingestion started!")
                return "Document ingestion started in background. Check back in a few minutes."
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        return [], "", "", ""


def create_gradio_app() -> gr.Blocks:
    """Create the Gradio application."""
    
    ui = AgenticRAGUI()
    
    with gr.Blocks(
        title="Agentic RAG System",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        )
    ) as app:
        gr.Markdown("""
        # 🔍 Agentic RAG System
        
        A production-ready Retrieval-Augmented Generation system with:
        - **Hybrid Retrieval** (BM25 + Vector Search)
        - **Self-Querying** metadata filtering
        - **Agentic Logic** using LangGraph
        - **Hallucination Guardrails**
        - **Observability** (token usage, latency, precision tracking)
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main query interface
                with gr.Group():
                    question_input = gr.Textbox(
                        label="Ask a Question",
                        placeholder="Enter your question here...",
                        lines=3,
                        scale=2
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🔍 Ask", variant="primary", scale=1)
                        clear_btn = gr.Button("🗑️ Clear", scale=0)
                
                # Show sources toggle
                show_sources = gr.Checkbox(
                    label="Show Sources",
                    value=True,
                    scale=0
                )
                
                # Response display
                answer_output = gr.Markdown(
                    label="Answer",
                    value="*Submit a question to get started*"
                )
            
            with gr.Column(scale=1):
                # Status and stats panel
                with gr.Group():
                    gr.Markdown("### 📊 Status")
                    
                    api_status = gr.Markdown("*Checking API status...*")
                    refresh_status_btn = gr.Button("🔄 Refresh Status")
                    
                    doc_stats = gr.Markdown("*Loading document stats...*")
                    refresh_docs_btn = gr.Button("📄 Refresh Docs")
                
                with gr.Group():
                    gr.Markdown("### ⚙️ Actions")
                    ingest_btn = gr.Button("📥 Ingest Documents")
                    ingest_status = gr.Textbox(label="Ingestion Status", interactive=False)
                
                with gr.Group():
                    gr.Markdown("### 📈 Query Stats")
                    stats_output = gr.Markdown("*No queries yet*")
        
        # Sources panel
        with gr.Accordion("📚 Sources & Context", open=False):
            sources_output = gr.Markdown("*No sources available*")
        
        # Chat history
        with gr.Accordion("💬 Chat History", open=False):
            chat_output = gr.Chatbot(label="Conversation History")
        
        # Event handlers
        def on_submit(question, show_src):
            answer, sources, stats = ui.query(question, show_src)
            history = [(q, a) for q, a in zip(ui.history[::2], ui.history[1::2][::2])]
            return answer, sources, stats, gr.update(value=history)
        
        submit_btn.click(
            fn=on_submit,
            inputs=[question_input, show_sources],
            outputs=[answer_output, sources_output, stats_output, chat_output]
        )
        
        question_input.submit(
            fn=on_submit,
            inputs=[question_input, show_sources],
            outputs=[answer_output, sources_output, stats_output, chat_output]
        )
        
        clear_btn.click(
            fn=ui.clear_history,
            inputs=[],
            outputs=[chat_output, answer_output, sources_output, stats_output]
        )
        
        refresh_status_btn.click(
            fn=ui.check_api_health,
            inputs=[],
            outputs=[api_status]
        )
        
        refresh_docs_btn.click(
            fn=ui.get_document_stats,
            inputs=[],
            outputs=[doc_stats]
        )
        
        ingest_btn.click(
            fn=ui.ingest_documents,
            inputs=[],
            outputs=[ingest_status]
        )
        
        # Initial status check
        app.load(fn=ui.check_api_health, inputs=[], outputs=[api_status])
        app.load(fn=ui.get_document_stats, inputs=[], outputs=[doc_stats])
        app.load(fn=ui.get_system_stats, inputs=[], outputs=[stats_output])
    
    return app


def main():
    """Main entry point for the Gradio app."""
    app = create_gradio_app()
    
    # Get port from environment or default
    port = int(os.getenv("GRADIO_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share
    )


if __name__ == "__main__":
    main()

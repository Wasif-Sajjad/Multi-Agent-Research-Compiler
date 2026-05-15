import streamlit as st
import os
import sys
import json
import requests
from dotenv import load_dotenv
from markdown_pdf import MarkdownPdf, Section

# Load environment variables from .env
load_dotenv()

# Project Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []
if "current_report" not in st.session_state:
    st.session_state.current_report = None

# Set up page configuration for an elegant, minimal look
st.set_page_config(
    page_title="Autonomous Research Compiler",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium, dynamic, and modern dark-mode aesthetic
st.markdown("""
    <style>
    /* Premium dark gradient background */
    .stApp {
        background-color: #0f172a;
        background-image: radial-gradient(circle at 50% 0%, #1e1b4b, #0f172a 80%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Ensure all default text is visible */
    .stMarkdown p, .stMarkdown li {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Dynamic Gradient Heading */
    h1 {
        background: linear-gradient(45deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #f8fafc !important;
        font-weight: 600;
    }
    
    /* Interactive Button */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
        background: linear-gradient(90deg, #60a5fa, #a78bfa) !important;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 1px #8b5cf6 !important;
    }
    
    /* Elegant blockquotes for sub-tasks */
    blockquote {
        border-left: 4px solid #8b5cf6 !important;
        background: rgba(30, 41, 59, 0.6) !important;
        padding: 1rem 1.5rem !important;
        border-radius: 0 8px 8px 0 !important;
        color: #e2e8f0 !important;
        margin: 1rem 0 !important;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.98) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        margin-bottom: 1rem !important;
    }

    [data-testid="stSidebar"] h3 {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px dashed rgba(139, 92, 246, 0.4) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Sidebar Ghost Buttons (History) */
    [data-testid="stSidebar"] .stButton>button {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: none !important;
        justify-content: flex-start !important;
        padding: 0.6rem 1rem !important;
        font-weight: 400 !important;
        width: 100% !important;
        color: #cbd5e1 !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: rgba(139, 92, 246, 0.15) !important;
        border-color: rgba(139, 92, 246, 0.4) !important;
        color: #ffffff !important;
        transform: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Autonomous Research Compiler")
st.markdown("Enter a research topic below. The system will autonomously decompose the query, search the web, read local documents, evaluate findings, and compile a fully cited report.")

# Helper functions for API interaction
def get_backend_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception:
        return []
    return []

def upload_to_backend(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return False

def delete_from_backend(filename):
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{filename}", timeout=10)
        return response.status_code == 200, response.json().get("chunks_removed", 0)
    except Exception as e:
        return False, str(e)

# Sidebar for configuration
with st.sidebar:
    st.markdown("<h2>Control Panel</h2>", unsafe_allow_html=True)
    
    st.markdown("### Settings")
    max_iterations = st.slider(
        "Max Research Loops", 
        min_value=1, max_value=5, value=3, 
        help="How many times the Critic can reject the findings and force more research."
    )
    
    st.markdown("### Knowledge Base")
    
    uploaded_files = st.file_uploader("Upload PDF Documents", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        success_count = 0
        for uf in uploaded_files:
            if upload_to_backend(uf):
                success_count += 1
        if success_count > 0:
            st.success(f"Successfully uploaded {success_count} file(s) to backend!")
            st.rerun()
        
    pdfs = get_backend_documents()
    if pdfs:
        with st.expander(f"View {len(pdfs)} Loaded PDFs"):
            for pdf in pdfs:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.caption(f"📄 {pdf}")
                with col2:
                    if st.button("Remove", key=f"del_{pdf}", help=f"Delete {pdf}"):
                        success, detail = delete_from_backend(pdf)
                        if success:
                            st.toast(f"Removed {pdf} and {detail} vector chunks.")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete: {detail}")

    else:
        st.caption("No local PDFs found in backend. Relying strictly on web search.")

    st.markdown("---")
    if st.button("🚨 Reset Database", use_container_width=True, help="Wipe all embeddings and local PDFs."):
        try:
            resp = requests.post(f"{API_BASE_URL}/reset", timeout=10)
            if resp.status_code == 200:
                st.toast("Database wiped successfully.")
                st.rerun()
            else:
                st.error("Reset failed on backend.")
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")

    st.markdown("### Chat History")
    if not st.session_state.history:
        st.caption("No previous research runs in this session.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            query_preview = item['query'][:30] + "..." if len(item['query']) > 30 else item['query']
            if st.button(f"🔍 {query_preview}", key=f"hist_{i}"):
                st.session_state.current_report = item

# Main search interface
query = st.text_input("Research Topic", placeholder="e.g. Compare the latest advancements in solid-state batteries vs lithium-ion...")

if st.button("Start Research", type="primary") and query:
    
    payload = {
        "query": query,
        "max_iterations": max_iterations
    }

    st.markdown("### Live Execution Log")
    
    final_report = ""
    sources = []
    
    with st.status("Connecting to backend...", expanded=True) as status:
        try:
            # Call the streaming research endpoint
            with requests.post(f"{API_BASE_URL}/research/stream", json=payload, stream=True, timeout=120) as r:
                if r.status_code != 200:
                    st.error(f"Backend returned error: {r.status_code}")
                    status.update(label="Connection Failed", state="error")
                else:
                    for line in r.iter_lines():
                        if line:
                            output = json.loads(line)
                            
                            # Check for errors in the stream
                            if "error" in output:
                                st.error(f"Agent Error: {output['error']}")
                                break

                            # Process node updates
                            for node_name, state_update in output.items():
                                if node_name == "orchestrator_node":
                                    status.update(label="Orchestrator is planning tasks...")
                                    tasks = state_update.get('sub_tasks', [])
                                    st.write(f"**Orchestrator**: Decomposed query into {len(tasks)} parallel sub-tasks.")
                                    for t in tasks:
                                        st.markdown(f"> - **[{t.get('source_type', 'web').upper()}]** {t.get('question')}")
                                            
                                elif node_name == "web_search_node":
                                    status.update(label="Web Search Agent is gathering data...")
                                    findings = state_update.get('web_findings', [])
                                    st.write(f"**Web Search Agent**: Extracted {len(findings)} data chunks from the internet.")
                                    
                                elif node_name == "pdf_reader_node":
                                    status.update(label="PDF Reader Agent is reading documents...")
                                    findings = state_update.get('pdf_findings', [])
                                    if findings:
                                        st.write(f"**PDF Reader Agent**: Processed and embedded local documents.")
                                        
                                elif node_name == "aggregator_node":
                                    iteration = state_update.get('iteration_count', 0)
                                    st.write(f"**Aggregator**: Merged findings for Research Loop {iteration}.")
                                    
                                elif node_name == "critic_node":
                                    status.update(label="Critic is evaluating research quality...")
                                    score = state_update.get('critic_score', 0.0)
                                    suff = state_update.get('critic_sufficient', False)
                                    feedback = state_update.get('critic_feedback', '')
                                    
                                    color = "green" if suff else "red"
                                    st.write(f"**Critic**: Scored findings at :{color}[{score:.2f} / 1.0].")
                                    
                                    if not suff:
                                        st.warning(f"**Critic Feedback:** {feedback}")
                                            
                                elif node_name == "synthesis_node":
                                    status.update(label="Synthesis Agent is writing the final report...")
                                    st.write("**Synthesis Agent**: Compiling citations and formatting markdown report.")
                                    final_report = state_update.get("final_report", "")
                                    sources = state_update.get("sources", [])

            status.update(label="Research Complete", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Failed to communicate with backend: {e}")
            status.update(label="Connection Lost", state="error")

    if final_report:
        st.session_state.current_report = {
            "query": query,
            "report": final_report,
            "sources": sources
        }
        st.session_state.history.append(st.session_state.current_report)

# Display the active report (either just finished or clicked from history)
if st.session_state.current_report:
    report_data = st.session_state.current_report
    r_query = report_data["query"]
    r_report = report_data["report"]
    r_sources = report_data["sources"]
    
    st.markdown("---")
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.header("Final Research Report")
    with col2:
        # Generate the PDF file dynamically
        try:
            pdf = MarkdownPdf()
            pdf.add_section(Section(r_report))
            pdf.save("temp_report.pdf")
            
            with open("temp_report.pdf", "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"Research_{r_query[:15].replace(' ', '_')}.pdf",
                mime='application/pdf'
            )
        except Exception as e:
            st.error("Could not generate PDF.")

    with st.container(border=True):
        st.markdown(r_report)
        
        if r_sources:
            st.markdown("---")
            st.subheader("References")
            for i, src in enumerate(r_sources, 1):
                if str(src).startswith("http"):
                    st.markdown(f"{i}. [{src}]({src})")
                else:
                    st.markdown(f"{i}. {src}")

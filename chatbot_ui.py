import streamlit as st
import requests
import json
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Swedish Legal RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #1f4e79;
    background-color: #f8f9fa;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.bot-message {
    background-color: #f1f8e9;
    border-left: 4px solid #4caf50;
}
.source-box {
    background-color: #fff3e0;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin: 0.5rem 0;
    border-left: 3px solid #ff9800;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Header
st.markdown('<h1 class="main-header">⚖️ Swedish Legal RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API URL configuration
    api_url = st.text_input(
        "RAG API URL", 
        value=st.session_state.api_url,
        help="URL of your RAG API server"
    )
    st.session_state.api_url = api_url
    
    # Query settings
    max_results = st.slider(
        "Max Results", 
        min_value=1, 
        max_value=10, 
        value=3,
        help="Number of documents to retrieve"
    )
    
    include_sources = st.checkbox(
        "Include Sources", 
        value=True,
        help="Show source documents with citations"
    )
    
    # API Health Check
    st.subheader("🏥 API Status")
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                st.success("✅ API is healthy!")
                st.json(health_data)
            else:
                st.error(f"❌ API Error: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat with Swedish Legal Documents")
    
    # Query input form
    with st.form(key="query_form", clear_on_submit=True):
        user_query = st.text_input(
            "Ask a question about Swedish law:",
            placeholder="e.g., Vad säger lagen om skatt på naturgrus?",
            disabled=st.session_state.processing
        )
        
        submit_button = st.form_submit_button(
            "🔍 Ask Question", 
            type="primary",
            disabled=st.session_state.processing
        )
    
    # Process query when form is submitted
    if submit_button and user_query.strip():
        st.session_state.processing = True
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "type": "user",
            "message": user_query,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Show loading spinner
        with st.spinner("🤔 Thinking..."):
            try:
                # Make API request
                payload = {
                    "query": user_query,
                    "max_results": max_results,
                    "include_sources": include_sources
                }
                
                response = requests.post(
                    f"{api_url}/query",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add bot response to chat history
                    st.session_state.chat_history.append({
                        "type": "bot",
                        "message": result["answer"],
                        "sources": result.get("sources", []),
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "model": result.get("model_used", "Unknown")
                    })
                    
                    st.success("✅ Response received!")
                    
                else:
                    error_msg = f"API Error: {response.status_code} - {response.text}"
                    st.error(f"❌ {error_msg}")
                    
                    # Add error to chat history
                    st.session_state.chat_history.append({
                        "type": "bot",
                        "message": f"Sorry, I encountered an error: {error_msg}",
                        "sources": [],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "model": "Error"
                    })
                        
            except Exception as e:
                error_msg = f"Connection error: {str(e)}"
                st.error(f"❌ {error_msg}")
                
                # Add error to chat history
                st.session_state.chat_history.append({
                    "type": "bot",
                    "message": f"Sorry, I couldn't connect to the API: {error_msg}",
                    "sources": [],
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "model": "Error"
                })
        
        st.session_state.processing = False
    
    # Display chat history
    st.subheader("📜 Chat History")
    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            if chat["type"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You ({chat['timestamp']}):</strong><br>
                    {chat['message']}
                </div>
                """, unsafe_allow_html=True)
                
            else:  # bot message
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Legal Assistant ({chat['timestamp']}):</strong><br>
                    {chat['message']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if chat.get("sources") and include_sources and len(chat["sources"]) > 0:
                    st.markdown("**📚 Sources:**")
                    for j, source in enumerate(chat["sources"], 1):
                        chunk_info = f" ({source['chunk_info']})" if source.get('chunk_info') else ""
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>{j}. {source['title']}{chunk_info}</strong><br>
                            📋 SFS: {source['sfs_number']} | 
                            🎯 Similarity: {source['similarity_score']}<br>
                            🔗 <a href="{source['url']}" target="_blank">Document</a> | 
                            <a href="{source['source_link']}" target="_blank">Source</a> | 
                            <a href="{source['amendment_register_link']}" target="_blank">Amendments</a>
                        </div>
                        """, unsafe_allow_html=True)
                
                if chat.get('model') != 'Error':
                    st.markdown(f"*Model: {chat.get('model', 'Unknown')}*")
            
            st.markdown("---")
    else:
        st.info("👋 Welcome! Ask a question about Swedish legal documents to get started.")

# with col2:
#     st.subheader("📊 Quick Stats")
    
#     # Display API stats
#     try:
#         response = requests.get(f"{api_url}/stats", timeout=5)
#         if response.status_code == 200:
#             stats = response.json()
#             st.metric("Total Documents", stats.get("total_documents", "N/A"))
#             st.metric("Embedding Model", stats.get("embedding_model", "N/A"))
#             st.metric("Chat Model", stats.get("chat_model", "N/A"))
            
#             if stats.get("sample_document"):
#                 st.subheader("📄 Sample Document")
#                 sample = stats["sample_document"]
#                 st.write(f"**Title:** {sample.get('title', 'N/A')}")
#                 st.write(f"**SFS:** {sample.get('sfs_number', 'N/A')}")
#                 st.write(f"**Chunked:** {sample.get('is_chunked', 'N/A')}")
#         else:
#             st.error("Failed to load stats")
#     except:
#         st.warning("API not available")
    
#     # Example queries
#     st.subheader("💡 Example Queries")
#     example_queries = [
#         "Vad säger lagen om skatt på naturgrus?",
#         "Vilka regler gäller för bygglov?",
#         "Hur fungerar arbetslöshetsersättning?",
#         "Vad är reglerna för bilskatt?",
#         "Vilka krav finns på miljötillstånd?"
#     ]
    
#     for i, query in enumerate(example_queries):
#         if st.button(f"📝 {query[:30]}...", key=f"example_{i}", disabled=st.session_state.processing):
#             # Add the example query to chat and process it
#             st.session_state.chat_history.append({
#                 "type": "user",
#                 "message": query,
#                 "timestamp": datetime.now().strftime("%H:%M:%S")
#             })
            
#             # Process the example query
#             st.session_state.processing = True
            
#             with st.spinner("🤔 Processing example..."):
#                 try:
#                     payload = {
#                         "query": query,
#                         "max_results": max_results,
#                         "include_sources": include_sources
#                     }
                    
#                     response = requests.post(
#                         f"{api_url}/query",
#                         json=payload,
#                         timeout=30
#                     )
                    
#                     if response.status_code == 200:
#                         result = response.json()
                        
#                         st.session_state.chat_history.append({
#                             "type": "bot",
#                             "message": result["answer"],
#                             "sources": result.get("sources", []),
#                             "timestamp": datetime.now().strftime("%H:%M:%S"),
#                             "model": result.get("model_used", "Unknown")
#                         })
                        
#                         st.success("✅ Example processed!")
                        
#                 except Exception as e:
#                     st.error(f"❌ Error processing example: {str(e)}")
            
#             st.session_state.processing = False
#             st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    ⚖️ Swedish Legal RAG Chatbot | Powered by Gemini & ChromaDB
</div>
""", unsafe_allow_html=True)

# Show processing status
if st.session_state.processing:
    st.info("🔄 Processing your request...")
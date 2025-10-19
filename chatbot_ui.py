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
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 3px solid #ff9800;
    font-size: 0.9em;
}
.translation-info {
    background-color: #e8f5e9;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin: 0.5rem 0;
    border-left: 3px solid #4caf50;
    font-size: 0.85em;
    color: #2e7d32;
}
.source-counter {
    text-align: center;
    color: #666;
    font-size: 0.9em;
    margin: 0.5rem 0;
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
if 'english_mode' not in st.session_state:
    st.session_state.english_mode = False
if 'source_indices' not in st.session_state:
    st.session_state.source_indices = {}  # Track current source index for each chat message

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
    
    # English mode toggle
    st.subheader("🌐 Language Settings")
    english_mode = st.toggle(
        "Enable English Mode",
        value=st.session_state.english_mode,
        help="When enabled, you can ask questions in English and receive answers in English"
    )
    st.session_state.english_mode = english_mode
    
    if english_mode:
        st.info("🇬🇧 English Mode: Ask questions in English, get answers in English")
    else:
        st.info("🇸🇪 Swedish Mode: Ask questions in Swedish, get answers in Swedish")
    
    # Query settings
    st.subheader("⚙️ Query Settings")
    max_results = st.slider(
        "Max Results", 
        min_value=1, 
        max_value=50, 
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
        st.session_state.source_indices = {}
        st.success("Chat history cleared!")

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat with Swedish Legal Documents")
    
    # Query input form
    with st.form(key="query_form", clear_on_submit=True):
        # Dynamic placeholder based on language mode
        if english_mode:
            placeholder_text = "e.g., What is the Act on inventory of goods for income taxation about?"
        else:
            placeholder_text = "e.g., Vad säger lagen om skatt på naturgrus?"
        
        user_query = st.text_input(
            "Ask a question about Swedish law:",
            placeholder=placeholder_text,
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
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "english_mode": english_mode
        })
        
        # Show loading spinner
        with st.spinner("🤔 Thinking..."):
            try:
                # Make API request
                payload = {
                    "query": user_query,
                    "max_results": max_results,
                    "include_sources": include_sources,
                    "english_mode": english_mode
                }
                
                response = requests.post(
                    f"{api_url}/query",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add bot response to chat history
                    st.session_state.chat_history.append({
                        "type": "bot",
                        "message": result["answer"],
                        "sources": result.get("sources", []),
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "model": result.get("model_used", "Unknown"),
                        "original_query": result.get("original_query"),
                        "translated_query": result.get("translated_query"),
                        "english_mode": english_mode
                    })
                    
                    # Initialize source index for this message
                    chat_idx = len(st.session_state.chat_history) - 1
                    st.session_state.source_indices[chat_idx] = 0
                    
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
                        "model": "Error",
                        "english_mode": english_mode
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
                    "model": "Error",
                    "english_mode": english_mode
                })
        
        st.session_state.processing = False
    
    # Display chat history
    st.subheader("📜 Chat History")
    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            actual_idx = len(st.session_state.chat_history) - 1 - i
            
            if chat["type"] == "user":
                mode_indicator = "🇬🇧" if chat.get('english_mode') else "🇸🇪"
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>{mode_indicator} You ({chat['timestamp']}):</strong><br>
                    {chat['message']}
                </div>
                """, unsafe_allow_html=True)
                
            else:  # bot message
                mode_indicator = "🇬🇧" if chat.get('english_mode') else "🇸🇪"
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>{mode_indicator} Legal Assistant ({chat['timestamp']}):</strong><br>
                    {chat['message']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show translation info if in English mode
                if chat.get('english_mode') and chat.get('translated_query'):
                    st.markdown(f"""
                    <div class="translation-info">
                        🔄 <strong>Translation:</strong> Query was translated to Swedish for search<br>
                        <em>Swedish query: "{chat['translated_query']}"</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show sources with navigation if available
                if chat.get("sources") and include_sources and len(chat["sources"]) > 0:
                    sources = chat["sources"]
                    
                    # Initialize source index if not exists
                    if actual_idx not in st.session_state.source_indices:
                        st.session_state.source_indices[actual_idx] = 0
                    
                    current_idx = st.session_state.source_indices[actual_idx]
                    
                    # Ensure index is within bounds
                    if current_idx >= len(sources):
                        current_idx = 0
                        st.session_state.source_indices[actual_idx] = 0
                    
                    st.markdown("**📚 Sources:**")
                    
                    # Navigation controls
                    col_prev, col_counter, col_next = st.columns([1, 2, 1])
                    
                    with col_prev:
                        if st.button("⬅️ Previous", key=f"prev_{actual_idx}", disabled=(len(sources) <= 1)):
                            st.session_state.source_indices[actual_idx] = (current_idx - 1) % len(sources)
                            st.rerun()
                    
                    with col_counter:
                        st.markdown(f"""
                        <div class="source-counter">
                            <strong>Source {current_idx + 1} of {len(sources)}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_next:
                        if st.button("Next ➡️", key=f"next_{actual_idx}", disabled=(len(sources) <= 1)):
                            st.session_state.source_indices[actual_idx] = (current_idx + 1) % len(sources)
                            st.rerun()
                    
                    # Display current source
                    source = sources[current_idx]
                    chunk_info = f" ({source['chunk_info']})" if source.get('chunk_info') else ""
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>{source['title']}{chunk_info}</strong><br>
                        📋 SFS: {source['sfs_number']} | 
                        🎯 Similarity: {source['similarity_score']}<br>
                        🔗 <a href="{source['url']}" target="_blank">View Document</a>
                    </div>
                    """, unsafe_allow_html=True)
                
                if chat.get('model') != 'Error':
                    st.markdown(f"*Model: {chat.get('model', 'Unknown')}*")
            
            st.markdown("---")
    else:
        welcome_msg = "👋 Welcome! Ask a question about Swedish legal documents to get started."
        if english_mode:
            welcome_msg = "👋 Welcome! Ask questions in English about Swedish legal documents to get started."
        st.info(welcome_msg)

# Right column - Info panel
with col2:
    st.subheader("ℹ️ How it works")
    
    if english_mode:
        st.markdown("""
        **English Mode Active:**
        
        1. 🇬🇧 Ask your question in **English**
        2. 🔄 System translates to **Swedish**
        3. 🔍 Searches Swedish legal docs
        4. 📄 Retrieves relevant documents
        5. 🤖 Generates answer in Swedish
        6. 🔄 Translates answer to **English**
        7. ✅ You receive English response!
        
        **Example Questions:**
        - What is the tax law about gravel?
        - what does the law say about the discrimination against part-time employees 
        - explain the Act on pharmaceutical benefits ?
        """)
    else:
        st.markdown("""
        **Swedish Mode Active:**
        
        1. 🇸🇪 Ställ din fråga på **svenska**
        2. 🔍 Söker relevanta dokument
        3. 📄 Hämtar lagtexter
        4. 🤖 Genererar svar baserat på lagen
        5. ✅ Du får svar på svenska!
        
        **Exempel på frågor:**
        - Vad säger skattelagen om grus?
        - Vad säger lagen om diskriminering av deltidsanställda?
        - Förklara lagen om läkemedelsförmåner?
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    ⚖️ Swedish Legal RAG Chatbot | Powered by Gemini & ChromaDB | 🌐 Multilingual Support
</div>
""", unsafe_allow_html=True)

# Show processing status
if st.session_state.processing:
    st.info("🔄 Processing your request...")
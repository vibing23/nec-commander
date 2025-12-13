import streamlit as st
import os
import time
import re
import google.generativeai as genai
from pathlib import Path
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions

st.set_page_config(
    page_title="NEC Commander",
    page_icon="⚡",
    layout="centered"
)

st.markdown("""
<style>
    /* Force a modern system font stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    
    /* Main Header Styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
    }
    .main-header h1 {
        color: #00A3E0;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle */
    .subtitle {
        color: #8B949E;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Info Cards Container */
    .info-cards {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Info Card */
    .info-card {
        background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1rem;
        flex: 1;
    }
    .info-card-title {
        color: #00A3E0;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .info-card-content {
        color: #C9D1D9;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Scope Badge */
    .scope-badge {
        background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .scope-badge p {
        margin: 0;
        color: #00A3E0;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Prompt Label */
    .prompt-label {
        color: #8B949E;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Prompt Buttons */
    .stButton > button {
        border: 1px solid #30363D;
        background: #161B22;
        color: #C9D1D9;
        font-weight: 400;
        transition: all 0.2s ease;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
    }
    .stButton > button:hover {
        border-color: #00A3E0;
        background: #1C2128;
        color: #00A3E0;
    }
    
    /* Reset Button Styling */
    .reset-btn button {
        background: transparent !important;
        border: 1px solid #F85149 !important;
        color: #F85149 !important;
    }
    .reset-btn button:hover {
        background: #F8514920 !important;
        border-color: #F85149 !important;
    }
    
    /* Chat Message Styling */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    
    /* User Message Bubble */
    [data-testid="stChatMessage"][data-testid*="user"],
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
        background: #1E3A8A !important;
        border: none;
    }
    
    /* AI Message Bubble */
    [data-testid="stChatMessage"][data-testid*="assistant"],
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #262730 !important;
        border: 1px solid #30363D;
    }
    
    /* Chat Input */
    .stChatInput > div {
        border-radius: 12px;
        border: 1px solid #30363D;
        background: #161B22;
    }
    .stChatInput textarea {
        color: #FAFAFA;
    }
    
    /* Status Container */
    [data-testid="stStatus"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
    }
    
    /* Version Badge */
    .version-badge {
        display: inline-block;
        background: #00A3E020;
        color: #00A3E0;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    /* Disclaimer */
    .disclaimer {
        color: #8B949E;
        font-size: 0.8rem;
        padding: 0.75rem;
        background: #0E111720;
        border-radius: 8px;
        border-left: 3px solid #F0883E;
        margin-bottom: 1rem;
    }
    
    /* Footer Branding */
    .footer-branding {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #30363D;
    }
    .footer-branding a {
        color: #00A3E0;
        text-decoration: none;
    }
    .footer-branding a:hover {
        text-decoration: underline;
    }
    .footer-branding p {
        color: #C9D1D9;
        margin: 0 0 0.25rem 0;
        font-size: 0.9rem;
    }
    .footer-branding .copyright {
        color: #8B949E;
        font-size: 0.8rem;
        margin: 0;
    }
    
    /* Citation Badge */
    .citation {
        display: inline-block;
        background: #00A3E020;
        color: #00A3E0;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with ChromaDB and Gemini embeddings"""
    api_key = get_secrets()
    if not api_key:
        return None, None
    
    genai.configure(api_key=api_key)
    
    # Initialize ChromaDB (local, persistent)
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use Gemini for embeddings (free!)
    embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/embedding-001"
    )
    
    # Get or create collection
    try:
        collection = client.get_collection(
            name="nec_article_625",
            embedding_function=embedding_function
        )
    except:
        collection = client.create_collection(
            name="nec_article_625",
            embedding_function=embedding_function,
            metadata={"description": "NEC 2023 Article 625 - Electric Vehicle Power Transfer Systems"}
        )
    
    return collection, genai.GenerativeModel('gemini-pro')

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for better context"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < text_len:
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:  # At least 70% of chunk
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def index_document(collection, pdf_path):
    """Index the PDF document into ChromaDB"""
    if collection.count() > 0:
        return  # Already indexed
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    
    # Add chunks to collection with metadata
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_index": i, "source": "NEC_2023_Article_625"} for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

def search_relevant_context(collection, query, n_results=3):
    """Search for relevant context from the NEC document"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    contexts = []
    if results and results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            contexts.append({
                'text': doc,
                'source': metadata.get('source', 'NEC 2023'),
                'chunk': metadata.get('chunk_index', i)
            })
    
    return contexts

def get_secrets():
    """Load secrets from environment variables or st.secrets"""
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
    
    return api_key

def check_secrets():
    """Check if required secrets are available"""
    api_key = get_secrets()
    if not api_key:
        st.error("⚠️ Missing API Key")
        st.markdown("""
        ### Setup Required:
        
        1. **Get Free Gemini API Key:**
           - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
           - Click "Get API Key"
           - Copy the key
        
        2. **Add to Streamlit:**
           - Go to your Streamlit Cloud dashboard
           - Click on your app settings
           - Add secret: `GEMINI_API_KEY = "your_key_here"`
        
        3. **Upload NEC PDF:**
           - Place your NEC Article 625 PDF in the root directory
           - Name it: `nec_article_625.pdf`
        
        **Free Tier:** 15 requests/min, 1500 requests/day
        """)
        st.stop()
    return api_key

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False

def get_assistant_response(model, collection, user_message):
    """Send message to Gemini with RAG context"""
    
    with st.status("Processing your query...", expanded=True) as status:
        status.write("🔍 Searching Article 625...")
        time.sleep(0.3)
        
        try:
            # Search for relevant context
            contexts = search_relevant_context(collection, user_message, n_results=3)
            
            status.write("📖 Analyzing NEC requirements...")
            time.sleep(0.3)
            
            # Build prompt with context
            context_text = "\n\n".join([
                f"[Context {i+1} from {ctx['source']}]:\n{ctx['text']}" 
                for i, ctx in enumerate(contexts)
            ])
            
            system_prompt = f"""You are NEC Commander, an expert AI assistant specialized in the National Electrical Code (NEC) 2023, specifically Article 625 - Electric Vehicle Power Transfer Systems.

IMPORTANT: Base your answers ONLY on the provided context from the NEC document below. If the answer is not in the context, say so clearly.

Context from NEC 2023 Article 625:
{context_text}

Instructions:
1. Answer based ONLY on the provided NEC context above
2. Cite specific NEC sections when applicable (e.g., "Per NEC 625.41...")
3. Be precise and code-compliant
4. If the context doesn't contain the answer, state: "I don't have specific information about that in Article 625. Please consult the full NEC codebook or a licensed electrician."
5. Always remind users this is for educational purposes

User Question: {user_message}"""
            
            status.write("✅ Generating response...")
            
            # Generate response
            response = model.generate_content(system_prompt)
            
            status.write("✓ Complete")
            time.sleep(0.2)
            status.update(label="Complete", state="complete", expanded=False)
            
            # Add citations
            response_with_citations = response.text
            for i, ctx in enumerate(contexts):
                response_with_citations += f"\n\n📚 **Reference {i+1}**: NEC 2023 Article 625 (Section {ctx['chunk']})"
            
            return response_with_citations
            
        except Exception as e:
            status.update(label="Error occurred", state="error")
            return f"I encountered an error: {str(e)}. Please try again."

def main():
    api_key = check_secrets()
    
    initialize_session_state()
    
    # Initialize RAG system
    if not st.session_state.rag_initialized:
        with st.spinner("🚀 Initializing NEC Commander..."):
            collection, model = initialize_rag_system()
            
            # Check if PDF exists
            pdf_path = "nec_article_625.pdf"
            if not Path(pdf_path).exists():
                st.error(f"""
                ⚠️ **NEC PDF Not Found**
                
                Please upload your NEC Article 625 PDF:
                1. Name it: `nec_article_625.pdf`
                2. Place it in the root directory of your app
                3. Redeploy
                
                The app needs this document to provide accurate answers.
                """)
                st.stop()
            
            # Index the document
            with st.spinner("📚 Indexing NEC Article 625... (first time only)"):
                index_document(collection, pdf_path)
            
            st.session_state.collection = collection
            st.session_state.model = model
            st.session_state.rag_initialized = True
    
    collection = st.session_state.collection
    model = st.session_state.model
    
    st.markdown("""
        <div class="main-header">
            <h1>⚡ NEC Commander <span class="version-badge">RAG v0.2</span></h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <p class="subtitle">AI Assistant for NEC 2023 Article 625 — Electric Vehicle Power Transfer Systems</p>
    """, unsafe_allow_html=True)
    
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    
    if not st.session_state.messages and not st.session_state.pending_prompt:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="info-card">
                    <div class="info-card-title">Current Scope</div>
                    <div class="info-card-content">
                        📘 NEC 2023 Article 625<br>
                        🔌 Electric Vehicle Power Transfer
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="info-card">
                    <div class="info-card-title">Capabilities</div>
                    <div class="info-card-content">
                        • RAG-powered search<br>
                        • Cited NEC references
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="disclaimer">
                ⚠️ For educational use only. Always verify with a licensed electrician or the official NEC codebook.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="prompt-label">Common Questions</p>', unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("What breaker size for a 48A charger?", use_container_width=True):
                st.session_state.pending_prompt = "What is the breaker size for a 48A charger?"
                st.rerun()
            if st.button("Hardwired charger disconnect required?", use_container_width=True):
                st.session_state.pending_prompt = "Does a hardwired charger require a disconnect?"
                st.rerun()
        with btn_col2:
            if st.button("Wire size for an EV charger?", use_container_width=True):
                st.session_state.pending_prompt = "What wire size do I need for an EV charger?"
                st.rerun()
            if st.button("Garage ventilation requirements?", use_container_width=True):
                st.session_state.pending_prompt = "What are the ventilation requirements for EV charging in a garage?"
                st.rerun()
        
        st.markdown("")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    prompt = st.chat_input("Ask about the National Electrical Code...")
    
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    
    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response_text = get_assistant_response(
            model,
            collection,
            prompt
        )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text
        })
        
        with st.chat_message("assistant"):
            st.markdown(response_text)
    
    st.markdown("""
        <div class="footer-branding">
            <p>🛠️ Built by <a href="https://www.linkedin.com/in/besniksulmataj/" target="_blank">Besnik Sulmataj</a></p>
            <p class="copyright">© 2025 | AI x Energy | RAG-Powered with Gemini (Free)</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import streamlit as st
import os
import time
import google.generativeai as genai
from pathlib import Path
import PyPDF2

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
    
    /* Hide sidebar completely when not needed */
    [data-testid="stSidebar"][aria-expanded="false"] {display: none;}
    
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
    
    /* Setup Warning */
    .setup-warning {
        background: #1C1917;
        border: 2px solid #F0883E;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    .setup-warning h3 {
        color: #F0883E;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

def get_secrets():
    """Load secrets from environment variables or st.secrets"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
    return api_key

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < text_len:
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def get_embedding(text, model):
    """Get embedding using Gemini"""
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    import math
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (magnitude_a * magnitude_b)

def search_relevant_context(chunks_with_embeddings, query, model, n_results=3):
    """Search for relevant context using embeddings"""
    # Get query embedding
    query_result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = query_result['embedding']
    
    # Calculate similarities
    similarities = []
    for i, (chunk, embedding) in enumerate(chunks_with_embeddings):
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((i, similarity, chunk))
    
    # Sort by similarity and get top results
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    contexts = []
    for i, (idx, score, chunk) in enumerate(similarities[:n_results]):
        contexts.append({
            'text': chunk,
            'source': 'NEC 2023 Article 625',
            'chunk': idx,
            'score': score
        })
    
    return contexts

def get_assistant_response(model, chunks_with_embeddings, user_message):
    """Generate response with RAG"""
    with st.status("Processing your query...", expanded=True) as status:
        status.write("🔍 Searching Article 625...")
        time.sleep(0.3)
        
        try:
            contexts = search_relevant_context(chunks_with_embeddings, user_message, model, n_results=3)
            
            status.write("📖 Analyzing NEC requirements...")
            time.sleep(0.3)
            
            context_text = "\n\n".join([
                f"[Context {i+1} (relevance: {ctx['score']:.2f})]:\n{ctx['text']}" 
                for i, ctx in enumerate(contexts)
            ])
            
            system_prompt = f"""You are NEC Commander, an expert on NEC 2023 Article 625 - Electric Vehicle Power Transfer Systems.

CRITICAL: Base your answers ONLY on the provided context below. If the answer is not in the context, clearly state that.

Context from NEC 2023 Article 625:
{context_text}

Instructions:
1. Answer based ONLY on the provided context
2. Cite specific NEC sections when you see them (e.g., "Per NEC 625.41...")
3. Be precise and code-compliant
4. If context lacks the answer, say: "I don't have specific information about that in the provided Article 625 content. Please consult the full NEC codebook."
5. Always remind users this is for educational purposes and should be verified with a licensed electrician

User Question: {user_message}"""
            
            status.write("✅ Generating response...")
            
            response = model.generate_content(system_prompt)
            
            status.write("✓ Complete")
            time.sleep(0.2)
            status.update(label="Complete", state="complete", expanded=False)
            
            response_with_citations = response.text
            for i, ctx in enumerate(contexts):
                response_with_citations += f"\n\n📚 **Reference {i+1}**: NEC 2023 Article 625 (Relevance: {ctx['score']:.0%})"
            
            return response_with_citations
            
        except Exception as e:
            status.update(label="Error occurred", state="error")
            return f"Error: {str(e)}"

def initialize_session_state():
    """Initialize session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False

def main():
    initialize_session_state()
    
    api_key = get_secrets()
    
    # Check if we need setup
    if not api_key or not st.session_state.rag_ready:
        st.markdown("""
            <div class="main-header">
                <h1>⚡ NEC Commander <span class="version-badge">RAG v0.2</span></h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <p class="subtitle">AI Assistant for NEC 2023 Article 625 — Electric Vehicle Power Transfer Systems</p>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="setup-warning">
                <h3>🔧 One-Time Setup Required</h3>
                <p>Upload your NEC Article 625 PDF below to get started. This file is processed in-memory and never stored on our servers.</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not api_key:
            st.error("⚠️ Missing Gemini API Key")
            st.markdown("""
            ### Get Free API Key:
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Click "Get API Key"
            3. Add to Streamlit Secrets: `GEMINI_API_KEY = "your_key"`
            """)
            return
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # PDF Upload
        uploaded_file = st.file_uploader(
            "Upload NEC Article 625 PDF",
            type=['pdf'],
            help="Upload your NEC 2023 Article 625 PDF document"
        )
        
        if uploaded_file:
            with st.spinner("📚 Processing PDF... This may take 30-60 seconds..."):
                try:
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    
                    # Chunk text
                    st.write("Splitting document into chunks...")
                    chunks = chunk_text(text)
                    
                    # Create embeddings for each chunk
                    st.write(f"Creating embeddings for {len(chunks)} chunks...")
                    chunks_with_embeddings = []
                    
                    progress_bar = st.progress(0)
                    for i, chunk in enumerate(chunks):
                        embedding = get_embedding(chunk, model)
                        chunks_with_embeddings.append((chunk, embedding))
                        progress_bar.progress((i + 1) / len(chunks))
                    
                    # Save to session state
                    st.session_state.chunks_with_embeddings = chunks_with_embeddings
                    st.session_state.model = model
                    st.session_state.rag_ready = True
                    
                    st.success("✅ PDF processed successfully! You can now start asking questions.")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        
        return
    
    # Main app (after setup)
    chunks_with_embeddings = st.session_state.chunks_with_embeddings
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
            chunks_with_embeddings,
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

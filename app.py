# app.py

# --- Imports ---
import streamlit as st
import pandas as pd
import base64
import os
from datetime import datetime
from PyPDF2 import PdfReader

# LangChain Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# FIX for asyncio error in Streamlit
import nest_asyncio
nest_asyncio.apply()

# --- Page Configuration ---
st.set_page_config(page_title="PaperTalk", page_icon="📄")

# --- Default SVG Icons (Self-Contained) ---
USER_AVATAR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-user">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
    <circle cx="12" cy="7" r="4"></circle>
</svg>
"""
BOT_AVATAR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-cpu">
    <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
    <rect x="9" y="9" width="6" height="6"></rect>
    <line x1="9" y1="1" x2="9" y2="4"></line>
    <line x1="15" y1="1" x2="15" y2="4"></line>
    <line x1="9" y1="20" x2="9" y2="23"></line>
    <line x1="15" y1="20" x2="15" y2="23"></line>
    <line x1="20" y1="9" x2="23" y2="9"></line>
    <line x1="20" y1="14" x2="23" y2="14"></line>
    <line x1="1" y1="9" x2="4" y2="9"></line>
    <line x1="1" y1="14" x2="4" y2="14"></line>
</svg>
"""

# --- Core Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    """Creates and saves a FAISS vector store from text chunks."""
    if not text_chunks:
        st.error("Could not extract text from the PDF. The file might be empty or corrupted.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store. Please check your API key and network connection. Error: {e}")
        return None

def get_conversational_chain(api_key):
    """Creates the conversational QA chain with a specific prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_user_input(user_question, api_key):
    """Handles user input, processes it, and updates the conversation history."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.session_state.conversation_history.append({
            "question": user_question,
            "answer": response["output_text"],
            "sources": docs
        })
    except Exception as e:
        st.error(f"An error occurred while processing your question: {e}")

# --- UI and Display Functions ---

def display_chat_exchange(exchange):
    """Renders a single user/bot chat exchange with sources."""
    user_avatar_base64 = base64.b64encode(USER_AVATAR_SVG.encode('utf-8')).decode('utf-8')
    bot_avatar_base64 = base64.b64encode(BOT_AVATAR_SVG.encode('utf-8')).decode('utf-8')

    st.markdown(
        f"""
        <div class="chat-message user">
            <div class="avatar">
                <img src="data:image/svg+xml;base64,{user_avatar_base64}" alt="User Avatar">
            </div>
            <div class="message">{exchange['question']}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="data:image/svg+xml;base64,{bot_avatar_base64}" alt="Bot Avatar">
            </div>
            <div class="message">{exchange['answer']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("View Sources"):
        if exchange['sources']:
            for i, doc in enumerate(exchange['sources']):
                st.info(f"Source {i+1}:\n{doc.page_content[:500]}...")
        else:
            st.warning("No sources were found for this answer.")

def main():
    """Main function to run the Streamlit app."""
    # --- Page Styling ---
    st.markdown("""
    <style>
        .chat-message { padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; align-items: center; }
        .chat-message.user { background-color: #3f51b5; color: #ffffff; }
        .chat-message.bot { background-color: #607d8b; color: #ffffff; }
        .chat-message .avatar { width: 15%; display: flex; justify-content: center; }
        .chat-message .avatar img { max-width: 60px; max-height: 60px; border-radius: 50%; object-fit: cover; background-color: #ffffff; padding: 5px; border: 1px solid #ddd; }
        .chat-message .message { width: 85%; padding: 0 1.5rem; color: #fff; }
        .stButton>button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

    # --- Session State Initialization ---
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False

    # --- Sidebar ---
    with st.sidebar:
        st.title("📄 PaperTalk Menu")

        # CORRECTED SECTION FOR ROBUST API KEY HANDLING
        with st.expander("🔑 API Key Setup", expanded=not st.session_state.get("api_key")):
            try:
                # This will succeed on Streamlit Cloud if the secret is set
                st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
                st.success("API key loaded from secrets!", icon="✅")
            except:
                # This will happen locally if secrets.toml doesn't exist or key is missing
                api_key_input = st.text_input(
                    "Enter your Google API Key:",
                    type="password",
                    value=st.session_state.get("api_key", "")
                )
                if api_key_input:
                    st.session_state.api_key = api_key_input
                    st.success("API Key saved!", icon="✅")
                else:
                    st.warning("Please enter your API Key to proceed.")
        
        with st.expander("📁 Documents & History", expanded=True):
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
            if st.button("Submit & Process Documents"):
                if not st.session_state.api_key:
                    st.warning("Please enter your Google API Key first.")
                elif not pdf_docs:
                    st.warning("Please upload at least one PDF file.")
                else:
                    with st.spinner("Processing documents... This may take a moment."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        if get_vector_store(text_chunks, st.session_state.api_key):
                            st.session_state.pdf_processed = True
                            st.success("Documents processed successfully! You can now ask questions.")
                        else:
                            st.session_state.pdf_processed = False
            
            if st.session_state.conversation_history:
                history_data = [{"Question": ex['question'], "Answer": ex['answer']} for ex in st.session_state.conversation_history]
                df = pd.DataFrame(history_data)
                st.download_button(
                    label="Download Conversation History",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='papertalk_history.csv',
                    mime='text/csv',
                    help="Download your entire chat history as a CSV file."
                )
            else:
                st.info("Chat history will appear here once you start interacting.")

    # --- Main Chat Interface ---
    st.header("PaperTalk: Chat with Your Documents 💬")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        if not st.session_state.api_key:
            st.warning("Please enter your API key in the sidebar.")
        elif not st.session_state.pdf_processed:
            st.warning("Please upload and process your documents first.")
        else:
            process_user_input(user_question, st.session_state.api_key)

    if st.session_state.conversation_history:
        for exchange in reversed(st.session_state.conversation_history):
            display_chat_exchange(exchange)
    else:
        st.info("👋 Welcome to PaperTalk! Upload your PDFs and enter your API key in the sidebar to get started.")

if __name__ == "__main__":
    main()
import streamlit as st
import sys
import pysqlite3
import os

sys.modules["sqlite3"] = pysqlite3

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Setup API Keys

# for .env setup
# from dotenv import load_dotenv
# load_dotenv()

# os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
# for deployment
hf_token = st.secrets["HF_TOKEN"]
os.environ["HF_TOKEN"] = hf_token

# Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# SIDEBAR 
st.sidebar.title("🔐 Configuration")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
session_id = st.sidebar.text_input("Session ID", value="default_session")
st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader("📄 Upload PDF(s)", type="pdf", accept_multiple_files=True)

# MAIN TITLE
st.title("📚 Chat with your PDFs")
st.caption("Ask questions, retrieve answers, and preserve chat history with Conversational RAG.")

# SESSION MANAGEMENT
if "store" not in st.session_state:
    st.session_state.store = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# MAIN FUNCTIONALITY
if api_key:
    if uploaded_files:
        with st.spinner("Processing uploaded PDFs..."):
            documents = []
            for uploaded_file in uploaded_files:
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = PyPDFLoader("temp.pdf")
                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory="./chroma_db")
            retriever = vectorstore.as_retriever()

        llm = ChatGroq(groq_api_key=api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

        # -------- PROMPTS & CHAINS --------
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, reformulate it into a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the context below.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # CHAT UI 
        st.markdown("### 💬 Chat Interface")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask a question about the PDFs...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            answer = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
    else:
        st.info("📎 Upload one or more PDF files to begin.")
else:
    st.warning("🚨 Please enter your Groq API key in the sidebar to continue.")

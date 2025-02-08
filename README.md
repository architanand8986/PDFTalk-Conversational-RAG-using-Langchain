# **Conversational RAG with PDF Uploads and Chat History**

This repository implements a **Conversational Retrieval-Augmented Generation (RAG) model** with **PDF document uploads** and **chat history tracking** using **Streamlit** for the web interface. It enables users to interact with uploaded PDFs, retrieve context-aware responses, and maintain session-based chat history.

## **Features**
- Upload multiple **PDF documents**
- Automatically **extract and index** content from PDFs
- Implement **retrieval-augmented generation (RAG)** for context-aware responses
- Maintain **chat history per session**
- Web interface powered by **Streamlit**
- Uses **LangChain**, **Hugging Face Embeddings**, and **ChromaDB** for retrieval
- Supports **Groq API** for language model inference

---

## **Setup Instructions**

### **1. Clone the repository**  
```bash
[git clone https://github.com/architanand8986/Conversational-RAG-with-PDF-uploads.git](https://github.com/architanand8986/Conversational-RAG-with-PDF-uploads.git)
```

### **2. Navigate to the project directory**  
```bash
cd Conversational-RAG-with-PDF-uploads
```

### **3. Create a virtual environment**  
- On **Windows** (cmd or PowerShell):  
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```  
- On **macOS/Linux**:  
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```  

### **4. Install dependencies**  
```bash
pip install -r requirements.txt
```

### **5. Set up environment variables**  
Create a `.env` file and add your **Groq API Key** and **Hugging Face Token**:
```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### **6. Run the Streamlit application**  
```bash
streamlit run app.py
```

---

## **Usage**
1. **Upload PDF files** via the web interface.
2. **Ask questions** about the uploaded content.
3. The system retrieves relevant context and generates **precise responses**.
4. **Chat history** is maintained per session.

---

## **Tech Stack**
- **Streamlit** - Web interface
- **LangChain** - Orchestrating retrieval and generation
- **Hugging Face Embeddings** - Text embedding model
- **ChromaDB** - Vector database for document retrieval
- **Groq API (Gemma2-9B-It)** - Large language model for response generation

---

## **License**
This project is licensed under the **MIT License**.

---

## **Contributing**
Contributions are welcome! Feel free to open issues and pull requests.

🚀 **Happy Building!**


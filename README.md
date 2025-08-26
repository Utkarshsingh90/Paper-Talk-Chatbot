# PaperTalk: Chat with Your Documents 📄💬

PaperTalk is an intelligent document chatbot built with Streamlit and Google's Generative AI. It allows you to upload multiple PDF files and ask questions about their content. The application extracts relevant information from the documents and provides detailed answers, citing the sources used.

---

## Key Features ✨

- **Multi-PDF Chat:** Upload one or more PDF documents to create a single knowledge base.
- **AI-Powered Q&A:** Leverages Google's gemini-1.5-flash model for fast and accurate answers.
- **Source Verification:** Each answer is accompanied by an expandable "View Sources" section showing the exact text chunks from the PDFs used.
- **Conversation History:** Keeps a complete log of your chat session.
- **Downloadable Chats:** Export your entire conversation history as a CSV file with one click.
- **Secure & Deployable:** Ready for production; secure API key handling via Streamlit secrets.

---

## Tech Stack 🛠️

- **Framework:** Streamlit
- **Language:** Python
- **LLM & Embeddings:** Google Generative AI (Gemini)
- **Core Libraries:** LangChain, PyPDF2, FAISS (vector storage)

---

## Getting Started

You can run PaperTalk locally on your machine:

### Prerequisites

- Python 3.9+
- Google Generative AI API Key ([Get one here](https://ai.google.dev/))

### Local Setup

### Clone the repository:
- git clone https://github.com/YOUR_USERNAME/papertalk-chatbot.git
- cd papertalk-chatbot

### Create and activate a virtual environment:

#### Windows
- python -m venv venv
- .\venv\Scripts\activate

#### macOS/Linux
-  python3 -m venv venv
- source venv/bin/activate

### Install required packages:
- pip install -r requirements.txt

### Run the Streamlit app:
- streamlit run app.py

The application will open in your web browser.

---
## Usage

1. Open the app in your browser (usually at `http://localhost:8501`).
2. In the sidebar, expand "API Key Setup" and enter your Google API Key.
3. Expand "Documents & History" and upload one or more PDF files.
4. Click "Submit & Process Documents" and wait for completion.
5. Use the main chat box to ask questions about your uploaded documents.

---

## Live Demo 🚀

Try the app instantly:
**[PaperTalk · Streamlit](https://paper-talk-chatbot.streamlit.app/)**

---
![PaperTalk Homepage](images/homepage.png)
 

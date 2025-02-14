# DocBot PDF Chatbot

**Chat with your PDFs!** DocBot is a Streamlit app that lets you ask questions about PDF documents. Upload a PDF or load from a URL, and DocBot uses AI to understand your questions, find answers, and summarize the information. Features include text-to-speech and a visual word cloud.

## Features

- **PDF Chat:**  Ask questions and get answers from PDF documents.
- **Upload or URL:** Load PDFs from your computer or directly from web URLs.
- **AI Summarization:**  Provides concise, AI-powered summaries for answers.
- **Contextual Answers:**  Finds the most relevant parts of the PDF to answer your questions.
- **Customizable Summaries:** Adjust summary length in the sidebar.
- **Relevance Control:**  Set how strictly DocBot finds relevant information.
- **Text-to-Speech:**  Hear chatbot responses aloud.
- **Word Cloud:**  See a visual overview of the PDF's main topics.

## AI Models

Powered by Hugging Face Transformers (Apache 2.0 License):

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` - for understanding text meaning.
- **Summarization:** `sshleifer/distilbart-cnn-12-6` - for creating summaries.

*(See code header for full license details)*

## Quick Start

1. **Load PDF:**
   - Upload a file or paste a PDF URL and load.
2. **Ask Questions:** Type your question and press Enter.
3. **View Answers:** Chat history shows your questions and DocBot's responses.
4. **Sidebar Settings:** Customize summary length, relevance, and text-to-speech.
5. **Word Cloud:**  Appears after PDF upload for a visual summary.

## Setup

**Dependencies:** Python 3.7+, and the libraries listed below.

```bash
pip install streamlit PyPDF2 sentence-transformers transformers langchain chromadb wordcloud matplotlib Pillow python-dotenv pyttsx3 requests numpy
```
pylint 8/10

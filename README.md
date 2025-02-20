# DocBot PDF Chatbot

**Chat with your PDFs!** DocBot is a rough-around-the-edges Streamlit app that lets you ask questions about PDF documents. 

Upload a PDF or load from a URL, and DocBot uses AI to understand your questions, find answers, and summarize the information. 

Features include text-to-speech and a visual word cloud.


[![N|Solid](https://user-images.githubusercontent.com/74038190/212280805-9bcb336b-8c55-46a8-abf8-ff286ab55472.gif)](https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub)

[![Generic badge](https://img.shields.io/badge/version-v0.5.0-<>.svg)](https://shields.io/)
[![PyPI license](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://pypi.python.org/pypi/ansicolortags/)

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
Live demo is online over at: https://huggingface.co/spaces/pajay/chatbot

Code tightness as per pylint 8/10

from itertools import chain

# DocBot PDF Chatbot: An application to chat with PDF documents.
# It allows users to upload PDFs, ask questions, and receive
# summarized answers, along with text-to-speech and word cloud features.

# Model License Information:
# --------------------------
# sentence-transformers/all-MiniLM-L6-v2: Apache 2.0 -
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# sshleifer/distilbart-cnn-12-6: Apache 2.0 -
# https://huggingface.co/sshleifer/distilbart-cnn-12-6
# --------------------------

import io
import os
import sys
import uuid
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyttsx3
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from wordcloud import WordCloud
import requests

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

CFG = {
    "transform": [("\t", " "), ("\n", " "), ("  ", " "), ("-", "")],
    "pdf_filter": [("PDF Files", "*.pdf")],
    "sel_prompt": "Select PDF Doc",
    "emb_model": "sentence-transformers/all-MiniLM-L6-v2",
    "db_coll": "doc_embeddings",
    "txt_chunk": 250,
    "txt_overlap": 125,
    "summ_model": "sshleifer/distilbart-cnn-12-6",
    "summ_max_w": 100,
    "summ_min_w": 30,
    "summ_txt_chunk": 2000,
    "search_limit": 10,
    "no_info_msg": "Sorry, no relevant info.",
    "relevance_threshold": 1.3,
    "enable_tts": True,
    "wordcloud_mask_path": None,
    "default_pdf_url": "https://www.warehouse-logistics.com/Download/Flyer/"
    "Brochure%20-%20Warehouse%20Management%20-%20Final.pdf",
}


def initialize_session_state():
    """Initializes Streamlit session state variables for the application."""
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "db_coll" not in st.session_state:
        st.session_state.db_coll = None
    if "summ_obj" not in st.session_state:
        st.session_state.summ_obj = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "wordcloud_image" not in st.session_state:
        st.session_state.wordcloud_image = None


client = chromadb.Client()


@st.cache_resource
def load_embedding_model(embedding_model_name):
    """Loads and caches the Sentence Transformer embedding model."""
    return SentenceTransformer(embedding_model_name)


@st.cache_resource
def load_summarizer_model(summ_model_name):
    """Loads and caches the summarization pipeline."""
    return pipeline("summarization", model=summ_model_name)


embedding_model = load_embedding_model(CFG["emb_model"])
summarizer = load_summarizer_model(CFG["summ_model"])

embedding_model = load_embedding_model(CFG["emb_model"])
summarizer = load_summarizer_model(CFG["summ_model"])


def transform_text(text, transformations=None):
    """Applies text transformations to normalize the input text.

    Args:
        text (str): Text to be transformed.
        transformations (list, optional): Transformations.

    Returns:
        str: Transformed lowercase text.
    """
    if transformations is None:
        transformations = CFG["transform"]
    if not transformations:
        return text
    if not isinstance(transformations, list):
        return text
    for old_text, replacement_text in transformations:
        text = text.replace(old_text, replacement_text)
    return text.lower()


def extract_text_from_pdf(pdf_path):
    """Extracts text content from each page of a PDF file.

    Args:
        pdf_path (str): File path to the PDF document.

    Returns:
        str: Combined text from PDF pages, lowercased.
    """
    text = "".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    return text.lower()


def create_text_chunks(
    text, chunk_size=CFG["txt_chunk"], chunk_overlap=CFG["txt_overlap"]
):
    """Splits large text into smaller, overlapping chunks for processing.

    Args:
        text (str): Input text to split.
        chunk_size (int, optional): Max chars per chunk.
        chunk_overlap (int, optional): Overlapping chars between chunks.

    Returns:
        list[str]: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


def generate_embeddings(text_chunks, embedding_model_name=CFG["emb_model"]):
    """Generates embeddings for text chunks using SentenceTransformer.

    Args:
        text_chunks (list[str]): List of text chunks for embedding.
        embedding_model_name (str, optional): Model name/path.

    Returns:
        tuple[list[list[float]], list[str]]: Embeddings and text chunks.
    """
    model = SentenceTransformer(embedding_model_name)
    return (
        model.encode([transform_text(chunk) for chunk in text_chunks]).tolist(),
        text_chunks,
    )


def create_embeddings_collection(collection_name=CFG["db_coll"]):
    """Creates or retrieves ChromaDB collection for storing embeddings.

    Args:
        collection_name (str, optional): ChromaDB collection name.

    Returns:
        chromadb.api.models.Collection.Collection or None: ChromaDB collection.
    """
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        try:
            return client.create_collection(name=collection_name)
        except Exception as e:
            st.error(f"Error creating embeddings collection: {e}")
            return None


def generate_wordcloud(text, mask_path=CFG["wordcloud_mask_path"]):
    """Generates word cloud image from text, optionally using a mask.

    Args:
        text (str): Text to generate word cloud from.
        mask_path (str, optional): Path to mask image file.

    Returns:
        PIL.Image.Image: Word cloud PIL Image object.
    """
    mask = np.array(Image.open(mask_path)) if mask_path else None
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        mask=mask,
        max_words=200,
        contour_width=3,
        contour_color="steelblue",
        collocations=False,
    ).generate(text)

    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return image


def load_document_and_embed(
    pdf_file_path, db_collection, embedding_model_name=CFG["emb_model"]
):
    """Loads PDF, generates embeddings, and adds to ChromaDB collection.

    Args:
        pdf_file_path (str): Path to the PDF file.
        db_collection (chromadb.api.models.Collection.Collection): ChromaDB collection.
        embedding_model_name (str, optional): Embedding model name.

    Returns:
        chromadb.api.models.Collection.Collection: ChromaDB collection.
    """
    if not pdf_file_path:
        raise ValueError("No document path provided.")
    pdf_text = extract_text_from_pdf(pdf_file_path)
    st.session_state.wordcloud_image = generate_wordcloud(pdf_text)
    text_chunks = create_text_chunks(pdf_text)
    embeddings, processed_chunks = generate_embeddings(
        text_chunks, embedding_model_name
    )
    ids = [str(uuid.uuid4()) for _ in range(len(processed_chunks))]
    metadatas = [{"text": chunk} for chunk in processed_chunks]
    if db_collection:
        db_collection.add(
            documents=processed_chunks,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )
    return db_collection


def fetch_relevant_context(
    query_str,
    db_collection,
    embedding_model_name=CFG["emb_model"],
    search_limit=CFG["search_limit"],
):
    """Fetches relevant document chunks from ChromaDB collection.

    Args:
        query_str (str): Query string for context search.
        db_collection (chromadb.api.models.Collection.Collection): ChromaDB collection.
        embedding_model_name (str, optional): Embedding model name.
        search_limit (int, optional): Max results to fetch.

    Returns:
        tuple[list[str], list[float]]: Relevant documents and distances.
    """
    if not db_collection:
        return None, None
    query_embedding = (
        SentenceTransformer(embedding_model_name).encode([query_str.lower()]).tolist()
    )
    results = db_collection.query(
        query_embeddings=query_embedding,
        n_results=search_limit,
        include=["documents", "distances"],
    )
    documents = results.get("documents")
    distances = results.get("distances")
    return documents, distances


def create_summary(
    context_segments,
    summarizer,
    min_words=CFG["summ_min_w"],
    max_words=CFG["summ_max_w"],
):
    """Generates summary from context segments using summarization pipeline.

    Args:
        context_segments (list[str]): Text segments to summarize.
        summarizer (transformers.pipelines.summarization.SummarizationPipeline): Summarizer pipeline.
        min_words (int, optional): Minimum words in summary.
        max_words (int, optional): Maximum words in summary.

    Returns:
        str or None: Generated summary, or None if no context.
    """
    if not context_segments:
        return None
    if isinstance(context_segments, list) and isinstance(context_segments[0], list):
        context_segments = list(chain.from_iterable(context_segments))
    context_text = " ".join(context_segments)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = summarizer(context_text, max_length=max_words, min_length=min_words)
    return output[0]["summary_text"].strip()


def text_to_speech(text, enable_text_to_speech):
    """Converts text to speech using pyttsx3, if enabled.

    Args:
        text (str): Text to be converted to speech.
        enable_text_to_speech (bool): Flag to enable/disable TTS.
    """
    if enable_text_to_speech:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


def load_pdf_from_url(url, db_collection):
    """Downloads PDF from URL, processes, and embeds into ChromaDB.

    Args:
        url (str): URL of the PDF document to load.
        db_collection (chromadb.api.models.Collection.Collection): ChromaDB collection.

    Returns:
        bool: True if PDF loaded and processed, False otherwise.
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        temp_pdf_path_url = "temp_url.pdf"
        with open(temp_pdf_path_url, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        load_document_and_embed(temp_pdf_path_url, db_collection)
        if os.path.exists(temp_pdf_path_url):
            os.remove(temp_pdf_path_url)
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading PDF from URL: {e}")
        return False
    except Exception as e:
        st.error(f"Error processing PDF from URL: {e}")
        return False


initialize_session_state()

st.title("📖 DocBot PDF Chatbot")

user_query = st.text_input("Ask questions about the PDF:", key="user_query")

min_summary_words = st.sidebar.slider("Min Summary Words", 10, 150, CFG["summ_min_w"])
max_summary_words = st.sidebar.slider("Max Summary Words", 50, 250, CFG["summ_max_w"])
relevance_threshold = st.sidebar.slider(
    "Relevance Threshold", 0.0, 1.0, CFG["relevance_threshold"]
)
enable_text_to_speech_checkbox = st.sidebar.checkbox(
    "Enable Text-to-Speech", CFG["enable_tts"]
)

CFG["summ_min_w"] = min_summary_words
CFG["summ_max_w"] = max_summary_words
CFG["relevance_threshold"] = relevance_threshold
CFG["enable_tts"] = enable_text_to_speech_checkbox


uploaded_pdf = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    on_change=lambda: st.session_state.pop("chat_history", None),
)

pdf_url = st.text_input("Or load PDF from URL:", value=CFG["default_pdf_url"])
load_url_button = st.button("Load PDF from URL")


if uploaded_pdf or load_url_button:
    if not st.session_state.pdf_uploaded:
        st.session_state.db_coll = create_embeddings_collection()
        st.session_state.summ_obj = pipeline("summarization", model=CFG["summ_model"])

        if not st.session_state.db_coll:
            st.error(
                "Failed to initialize document embeddings. Please check logs for errors."
            )
        pdf_loaded_success_flag = False
        if uploaded_pdf:
            try:
                temp_pdf_path_upload = "temp.pdf"
                with open(temp_pdf_path_upload, "wb") as f:
                    f.write(uploaded_pdf.getvalue())
                if st.session_state.pdf_uploaded and st.session_state.db_coll:
                    client.delete_collection(CFG["db_coll"])
                    st.session_state.db_coll = create_embeddings_collection()

                load_document_and_embed(temp_pdf_path_upload, st.session_state.db_coll)
                st.session_state.pdf_uploaded = True
                pdf_loaded_success_flag = True

            except Exception as e:
                st.error(f"Error loading PDF: {e}")
            finally:
                if os.path.exists(temp_pdf_path_upload):
                    os.remove(temp_pdf_path_upload)

        elif load_url_button:
            st.session_state.pop("chat_history", None)
            pdf_loaded_success_flag = load_pdf_from_url(
                pdf_url, st.session_state.db_coll
            )
            st.session_state.pdf_uploaded = pdf_loaded_success_flag

        if pdf_loaded_success_flag:
            st.success("PDF document loaded and processed successfully!")
            if st.session_state.wordcloud_image:
                st.image(
                    st.session_state.wordcloud_image,
                    caption="Word Cloud of Document Text",
                )


if st.session_state.pdf_uploaded and user_query:
    doc_context, search_distances = fetch_relevant_context(
        user_query, st.session_state.db_coll
    )
    if (
        search_distances
        and doc_context
        and min(search_distances[0]) < CFG["relevance_threshold"]
    ):
        bot_summary = create_summary(
            doc_context, st.session_state.summ_obj, CFG["summ_min_w"], CFG["summ_max_w"]
        )
        bot_response = bot_summary or CFG["no_info_msg"]
    else:
        bot_response = CFG["no_info_msg"]

    st.session_state.chat_history.append({"user": user_query, "bot": bot_response})

    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Chatbot:** {chat['bot']}")
    text_to_speech(bot_response, CFG["enable_tts"])

    with st.sidebar.expander("Debug Info"):
        st.write("Debug - Similarity Search Distances:")
        st.write(search_distances)
        if doc_context and doc_context[0]:
            st.write(
                "Debug - Retrieved Context (First Chunk if any):",
                doc_context[0][0][:100],
            )
        else:
            st.write("Debug - Retrieved Context: No context found or context is empty.")

    st.sidebar.markdown(
        """
            ---
            Created by [th3pajay](https://github.com/th3pajay)
            ![UserGIF](https://user-images.githubusercontent.com/74038190/219925470-37670a3b-c3e2-4af7-b468-673c6dd99d16.png)
        """
    )

else:
    st.info("Please upload a PDF document or load from URL to start chatting.")

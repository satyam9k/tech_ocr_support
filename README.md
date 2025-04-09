# Technical Document Assistant

This Streamlit application acts as an intelligent assistant for your technical documents (PDFs and Images). It leverages OCR, vector databases, and large language models (LLMs) to help you understand, query, and generate content based on your uploaded files.

## Features

*   **Document Upload:** Upload PDF and Image files.
*   **OCR Processing:** Uses Mistral AI's OCR service to extract text content from uploaded documents.
*   **Vector Storage:** Embeds document content using Sentence Transformers and stores it in a Chroma vector database for efficient similarity search.
*   **Topic Suggestions:** Analyzes processed documents using Google Gemini to suggest key topics and concepts for further exploration.
*   **Question Answering (RAG):** Answer questions based on the content of your documents using a Retrieval-Augmented Generation (RAG) approach with Google Gemini.
*   **Internet Search Integration:** Optionally augment answers and context by searching the web using the Google Custom Search API. Search results are cached in the vector database.
*   **Implementation Guidance:** Generate step-by-step implementation instructions for topics identified in your documents or suggested by the assistant, leveraging Google Gemini.
*   **Code Generation:** Generate code snippets based on the technical context found in your documents and specific user prompts, using Google Gemini.
*   **Chat & Code History:** Maintains a history of your questions and generated code for context and reference.

## Requirements

*   Python 3.8+
*   Pip (Python package installer)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/satyam9k/tech_ocr_support.git
    cd tech_ocr_support
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    python-dotenv
    mistralai
    langchain-community
    langchain
    google-generativeai
    requests
    pathlib
    chromadb
    sentence-transformers
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

This application requires API keys for several services.

1.  **Create a `.env` file** in the root directory of the project.
2.  **Add the following environment variables** to the `.env` file, replacing `<your_key>` and `<your_cx>` with your actual credentials:

    ```dotenv
    # Mistral AI API Key (for OCR)
    MISTRAL_API_KEY=<your_mistral_api_key>

    # Google AI API Key (for Gemini models)
    GOOGLE_API_KEY=<your_google_ai_api_key>

    # Google Custom Search API Key & CX ID (for internet search)
    # See: https://developers.google.com/custom-search/v1/overview
    GOOGLE_SEARCH_API_KEY=<your_google_search_api_key>
    GOOGLE_SEARCH_CX=<your_google_search_cx_id>
    ```

    *   **Mistral AI:** Get your API key from the Mistral AI platform.
    *   **Google AI (Gemini):** Get your API key from Google AI Studio.
    *   **Google Custom Search:** You'll need to set up a Custom Search Engine and get an API key from the Google Cloud Console. Follow the instructions here.

## Usage

1.  Ensure your virtual environment is activated and you are in the project directory.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your web browser.
4.  Use the sidebar to upload PDF or Image files.
5.  Click "Process Documents" to start OCR and vectorization.
6.  Navigate through the tabs ("Suggested Topics", "Ask Questions", "Code Generation") to interact with the assistant.

## Testing Google Search Integration

A separate script `int.py` is provided to test your Google Custom Search API credentials independently.

1.  Make sure your `.env` file is correctly configured with `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_CX`.
2.  Run the script from your terminal:
    ```bash
    python int.py
    ```
3.  Check the output for success or error messages regarding the API connection.




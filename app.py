import streamlit as st
import os
import tempfile
import json
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai import TextChunk
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import google.generativeai as genai
import requests
from pathlib import Path
import uuid
from datetime import datetime

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key)

gemini_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)

google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
google_search_cx = os.getenv("GOOGLE_SEARCH_CX")

# Initialize vector database
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
if not os.path.exists("db"):
    os.makedirs("db")
vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)

class PDFAssistant:
    def __init__(self):
        self.uploaded_files = []
        self.processed_content = {}
        self.document_context = []
        # Initialize code generation memory
        self.code_history = []
    
    def process_pdf_with_ocr(self, file_path, file_name):
        """Process PDF with Mistral OCR"""
        try:
            with open(file_path, "rb") as f:
                uploaded_pdf = mistral_client.files.upload(
                    file={
                        "file_name": file_name,
                        "content": f,
                    },
                    purpose="ocr"
                )
            
            signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)
            
            ocr_response = mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )
            
            # Extract markdown content from all pages
            all_markdown = ""
            for page in ocr_response.pages:
                all_markdown += f"{page.markdown}\n\n"
            
            return all_markdown
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    
    def process_image_with_ocr(self, file_path, file_name):
        """Process image with Mistral OCR"""
        try:
            with open(file_path, "rb") as f:
                uploaded_image = mistral_client.files.upload(
                    file={
                        "file_name": file_name,
                        "content": f,
                    },
                    purpose="ocr"
                )
            
            signed_url = mistral_client.files.get_signed_url(file_id=uploaded_image.id)
            
            ocr_response = mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": signed_url.url,
                }
            )
            
            # Extract markdown content
            all_markdown = ""
            for page in ocr_response.pages:
                all_markdown += f"{page.markdown}\n\n"
            
            return all_markdown
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    
    def store_in_vector_db(self, content, file_name):
        """Store content in vector database with improved chunking"""
        # Using recursive chunker to better handle text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=800,  # Reduced chunk size to avoid exceeding limits
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(content)
        
        # Create documents with metadata
        docs = [Document(page_content=chunk, metadata={"source": file_name}) for chunk in chunks]
        
        # Add to vector store
        vectorstore.add_documents(docs)
        vectorstore.persist()
        
        # Update document context
        self.document_context.extend(docs)
        
        return len(docs)
    
    def search_vector_db(self, query, k=5):
        """Search vector database for relevant documents"""
        results = vectorstore.similarity_search(query, k=k)
        return results
    
    def generate_content_suggestions(self):
        """Generate topic/concept suggestions from the uploaded documents"""
        if not self.document_context:
            return []
        
        # Combine some document content for context
        combined_content = "\n\n".join([doc.page_content for doc in self.document_context[:10]])
        
        # Use Gemini to generate suggestions
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Based on the following technical document content, suggest 5-7 key topics or concepts 
        that a user might want to implement or learn more about. Format as a JSON list of objects 
        with 'title' and 'description' fields.
        
        Document content:
        {combined_content}
        """
        
        response = model.generate_content(prompt)
        
        try:
            # Extract JSON from response
            suggestions_text = response.text
            # Find JSON in the response
            start = suggestions_text.find('[')
            end = suggestions_text.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = suggestions_text[start:end]
                suggestions = json.loads(json_str)
                return suggestions
            return []
        except Exception as e:
            st.error(f"Error parsing suggestions: {str(e)}")
            return []
    
    def search_internet(self, query):
        """
        Search the internet using Google Custom Search API and store results in vector database
        for future retrieval
        """
        try:
            # Create a unique identifier for this search
            search_id = f"web_search_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Check if we have cached results for similar queries
            similar_queries = self.search_vector_db(query, k=2)
            cached_results = []
            
            for doc in similar_queries:
                if doc.metadata.get('source', '').startswith('web_search_'):
                    cached_results.append(doc)
            
            # If we have relevant cached results and they're recent (less than 24 hours old)
            if cached_results:
                # Extract the timestamp from the source ID
                latest_cached = max(cached_results, key=lambda x: x.metadata.get('source', ''))
                time_str = latest_cached.metadata.get('source', '').replace('web_search_', '')
                if time_str:
                    try:
                        cached_time = datetime.strptime(time_str, '%Y%m%d%H%M%S')
                        time_diff = datetime.now() - cached_time
                        # If cache is less than 24 hours old, use it
                        if time_diff.total_seconds() < 86400:  # 24 hours in seconds
                            st.info("Using cached search results")
                            return json.loads(latest_cached.metadata.get('search_results', '[]'))
                    except:
                        pass
            
            # Perform the search with Google Custom Search API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': google_search_api_key,
                'cx': google_search_cx,
                'q': query,
                'num': 10  # Increased from 5 to 10 for better coverage
            }
            response = requests.get(url, params=params)
            results = response.json()
            
            # Extract relevant information
            search_results = []
            if 'items' in results:
                for item in results['items']:
                    result = {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', '')
                    }
                    
                    # Extract more content if available
                    if 'pagemap' in item and 'metatags' in item['pagemap']:
                        for metatag in item['pagemap']['metatags']:
                            if 'og:description' in metatag:
                                result['description'] = metatag['og:description']
                    
                    search_results.append(result)
            
            # Store search results in vector database for future use
            if search_results:
                # Combine all search results into one comprehensive document
                combined_content = f"Search query: {query}\n\n"
                for result in search_results:
                    combined_content += f"Title: {result['title']}\n"
                    combined_content += f"Link: {result['link']}\n"
                    combined_content += f"Snippet: {result['snippet']}\n"
                    if 'description' in result:
                        combined_content += f"Description: {result['description']}\n"
                    combined_content += "\n---\n\n"
                
                # Create a document with search results metadata
                doc = Document(
                    page_content=combined_content,
                    metadata={
                        "source": search_id,
                        "query": query,
                        "search_results": json.dumps(search_results),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                
                # Add to vector store
                vectorstore.add_documents([doc])
                vectorstore.persist()
                
                # Update document context
                self.document_context.append(doc)
            
            return search_results
        except Exception as e:
            st.error(f"Error searching internet: {str(e)}")
            return []

    def answer_with_rag(self, user_question, allow_internet=False):
        """Generate a comprehensive answer using RAG approach with both document and internet sources"""
        # First search vector DB for document-based context
        relevant_docs = self.search_vector_db(user_question)
        doc_context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Track source type for better response formatting
        sources = []
        if doc_context:
            sources.append("documents")
        
        # If internet search is allowed or no document context found
        internet_context = ""
        internet_results = []
        if allow_internet or not doc_context:
            internet_results = self.search_internet(user_question)
            if internet_results:
                internet_context = "\n\n".join([
                    f"Title: {result['title']}\nSnippet: {result['snippet']}" 
                    for result in internet_results
                ])
                sources.append("internet")
        
        # If we have no context from either source
        if not sources:
            return "I couldn't find information about this in your documents or the internet. Please try rephrasing your question or providing more context."
        
        # Combine contexts, but keep them distinguished
        combined_prompt = f"User question: {user_question}\n\n"
        
        if "documents" in sources:
            combined_prompt += f"Document context:\n{doc_context}\n\n"
        
        if "internet" in sources:
            combined_prompt += f"Internet search results:\n{internet_context}\n\n"
        
        # Generate comprehensive response with Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
    Based on the following information, provide a comprehensive answer to the user's question.
    
    {combined_prompt}
    
    Instructions:
    1. Synthesize information from all available sources to create a coherent, detailed response
    2. Clearly distinguish between information from documents versus internet sources
    3. For internet sources, identify the relevant source links where appropriate
    4. If information is insufficient or contradictory, acknowledge this
    5. Format your response in a readable, well-structured way with appropriate markdown
    6. Include relevant links from internet sources at the end of your response
    
    Your answer should be comprehensive but focused on addressing the specific question.
    """
        
        response = model.generate_content(prompt)
        
        # Format answer with clear indication of sources
        answer = response.text
        
        # If we used internet sources, append source links
        if "internet" in sources:
            # Check if the response already includes sources; if not, add them
            if not "Sources:" in answer:
                answer += "\n\n**Sources:**"
                for result in internet_results:
                    # Only include sources that are likely referenced in the answer
                    keywords = [word.lower() for word in user_question.split() if len(word) > 3]
                    snippet_lower = result['snippet'].lower()
                    if any(keyword in snippet_lower for keyword in keywords) or any(keyword in result['title'].lower() for keyword in keywords):
                        answer += f"\n- [{result['title']}]({result['link']})"
        
        return answer
    
    def generate_implementation_steps(self, topic, use_internet=False):
        """Generate implementation steps for a topic"""
        # Search vector DB for relevant context
        relevant_docs = self.search_vector_db(topic)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Add internet search results if requested
        internet_context = ""
        internet_sources = []
        if use_internet or not context:
            search_results = self.search_internet(topic)
            internet_context = "\n\n".join([f"Title: {result['title']}\nSnippet: {result['snippet']}" 
                                          for result in search_results])
            internet_sources = [{"title": result['title'], "url": result['link']} for result in search_results]
        
        # Craft prompt based on available context
        if context:
            source = "the provided documents"
            combined_context = context
            sources_list = []
        elif internet_context:
            source = "internet search results"
            combined_context = internet_context
            sources_list = internet_sources
        else:
            return {"error": "No relevant information found"}
        
        # Use Gemini to generate implementation steps
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Based on the following information from {source}, provide clear step-by-step instructions
        for implementing or understanding '{topic}'. 
        
        Include code snippets where appropriate, formatted with proper markdown code blocks.
        If diagrams or workflows are mentioned in the context, describe them clearly.
        
        Context information:
        {combined_context}
        
        Format your response as JSON with the following structure:
        {{
            "title": "How to Implement {topic}",
            "overview": "Brief overview of what this implementation involves",
            "steps": [
                {{
                    "step": "Step 1",
                    "description": "Detailed explanation",
                    "code": "Code snippet if applicable (or null if none)"
                }},
                ...
            ],
            "additional_notes": "Any additional important information",
            "source": "{source}",
            "sources_list": {json.dumps(sources_list) if internet_sources else "[]"}
        }}
        """
        
        response = model.generate_content(prompt)
        
        try:
            # Extract JSON from response
            result_text = response.text
            # Find JSON in the response
            start = result_text.find('{')
            end = result_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = result_text[start:end]
                steps = json.loads(json_str)
                # Store implementation in session state
                if "implementation_results" not in st.session_state:
                    st.session_state.implementation_results = {}
                implementation_id = str(uuid.uuid4())
                st.session_state.implementation_results[implementation_id] = {
                    "topic": topic,
                    "steps": steps,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return steps, implementation_id
            return {"error": "Failed to generate structured steps"}, None
        except Exception as e:
            st.error(f"Error generating steps: {str(e)}")
            return {"error": f"Error: {str(e)}"}, None
    
    def generate_code(self, prompt, context=None, previous_code=None):
        """Generate or debug code based on PDF content and user request"""
        # Get relevant context if not provided
        if not context:
            relevant_docs = self.search_vector_db(prompt)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Prepare prompt with history context if provided
        code_context = ""
        if previous_code:
            code_context = f"\nPrevious code version:\n```python\n{previous_code}\n```\n"
        
        # Use Gemini to generate code
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt_text = f"""
        Based on the following technical context and user request, generate appropriate code.
        
        Technical context:
        {context}
        {code_context}
        
        User request:
        {prompt}
        
        Provide a complete, working solution with explanatory comments.
        """
        
        response = model.generate_content(prompt_text)
        
        # Add to code history
        code_id = str(uuid.uuid4())
        self.code_history.append({
            "id": code_id,
            "prompt": prompt,
            "code": response.text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return response.text, code_id

# Streamlit UI
def main():
    st.set_page_config(page_title="Technical PDF Assistant", layout="wide")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = PDFAssistant()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'current_implementation' not in st.session_state:
        st.session_state.current_implementation = None
        
    if 'implementation_results' not in st.session_state:
        st.session_state.implementation_results = {}
        
    if 'view_implementation' not in st.session_state:
        st.session_state.view_implementation = False
    
    st.title("Technical PDF Assistant")
    
    # Sidebar for file uploads and processing
    with st.sidebar:
        st.header("Upload Documents")
        file_type = st.radio("Select file type", ["PDF", "Image"])
        
        if file_type == "PDF":
            uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        else:
            uploaded_files = st.file_uploader("Choose Image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            process_button = st.button("Process Documents")
            if process_button:
                with st.spinner(f"Processing {file_type}s with OCR and storing in vector database..."):
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        temp_dir = tempfile.mkdtemp()
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process with OCR
                        st.write(f"Processing {uploaded_file.name}...")
                        
                        if file_type == "PDF":
                            content = st.session_state.assistant.process_pdf_with_ocr(temp_path, uploaded_file.name)
                        else:
                            content = st.session_state.assistant.process_image_with_ocr(temp_path, uploaded_file.name)
                        
                        if content:
                            # Store in vector DB
                            chunks = st.session_state.assistant.store_in_vector_db(content, uploaded_file.name)
                            st.success(f"Processed {uploaded_file.name} and stored {chunks} chunks in vector database")
                            
                            # Store content
                            st.session_state.assistant.processed_content[uploaded_file.name] = content
                            if uploaded_file.name not in st.session_state.assistant.uploaded_files:
                                st.session_state.assistant.uploaded_files.append(uploaded_file.name)
        
        if st.session_state.assistant.uploaded_files:
            st.subheader("Processed Files")
            for file_name in st.session_state.assistant.uploaded_files:
                st.write(f"- {file_name}")
    
    # Handle implementation view state
    if st.session_state.view_implementation and st.session_state.current_implementation:
        # Display implementation results
        if st.session_state.current_implementation in st.session_state.implementation_results:
            impl_data = st.session_state.implementation_results[st.session_state.current_implementation]
            
            # Back button
            if st.button("← Back to Topics"):
                st.session_state.view_implementation = False
                st.rerun()
            
            st.header(f"Implementation: {impl_data['topic']}")
            steps = impl_data['steps']
            
            st.subheader(steps.get("title", "Implementation Steps"))
            st.write(steps.get("overview", ""))
            
            for step_num, step in enumerate(steps.get("steps", [])):
                st.markdown(f"### {step.get('step', f'Step {step_num+1}')}")
                st.write(step.get("description", ""))
                
                if step.get("code"):
                    st.code(step.get("code"), language="python")
            
            if steps.get("additional_notes"):
                st.markdown("### Additional Notes")
                st.write(steps.get("additional_notes"))
            
            st.info(f"Source: {steps.get('source', 'Document analysis')}")
            
            # Display source links if available
            if "sources_list" in steps and steps["sources_list"]:
                st.subheader("Sources")
                for source in steps["sources_list"]:
                    st.markdown(f"[{source['title']}]({source['url']})")
        else:
            st.error("Implementation not found")
            if st.button("Back to Topics"):
                st.session_state.view_implementation = False
                st.rerun()
    else:
        # Main area tabs
        tab1, tab2, tab3 = st.tabs(["Suggested Topics", "Ask Questions", "Code Generation"])
        
        # Tab 1: Suggested Topics
        with tab1:
            st.header("Suggested Topics from Your Documents")
            
            if st.session_state.assistant.uploaded_files:
                if st.button("Generate Topic Suggestions"):
                    with st.spinner("Analyzing documents and generating suggestions..."):
                        suggestions = st.session_state.assistant.generate_content_suggestions()
                        
                        if suggestions:
                            st.session_state.suggestions = suggestions
                        else:
                            st.warning("No topic suggestions could be generated. Try uploading more documents.")
                
                # Display suggestions if available
                if "suggestions" in st.session_state:
                    for i, suggestion in enumerate(st.session_state.suggestions):
                        with st.expander(f"{suggestion.get('title', f'Topic {i+1}')}"):
                            st.write(suggestion.get('description', 'No description available'))
                            
                            # Create a button with a unique key
                            button_key = f"impl_{suggestion.get('title', '')}_{i}"
                            if st.button(f"Implement '{suggestion.get('title', f'Topic {i+1}')}'", key=button_key):
                                with st.spinner("Generating implementation steps..."):
                                    steps, impl_id = st.session_state.assistant.generate_implementation_steps(
                                        suggestion.get('title', f'Topic {i+1}')
                                    )
                                    
                                    if isinstance(steps, dict) and "error" in steps:
                                        st.error(steps["error"])
                                        use_internet = st.checkbox("Search the internet for this topic?", key=f"inet_{i}")
                                        
                                        if use_internet and st.button("Search and Generate Steps", key=f"search_{i}"):
                                            with st.spinner("Searching internet and generating steps..."):
                                                steps, impl_id = st.session_state.assistant.generate_implementation_steps(
                                                    suggestion.get('title', f'Topic {i+1}'),
                                                    use_internet=True
                                                )
                                    
                                    if not (isinstance(steps, dict) and "error" in steps):
                                        st.session_state.current_implementation = impl_id
                                        st.session_state.view_implementation = True
                                        st.rerun()
            else:
                st.info("Upload and process documents to see topic suggestions")
        
        # Tab 2: Ask Questions
        with tab2:
            st.header("Ask Questions About Your Documents")
            
            if st.session_state.assistant.uploaded_files:
                # Display chat history
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**Assistant:** {message['content']}")
                
                # Input for new questions
                user_question = st.text_input("Ask a question about your documents:")
                col1, col2 = st.columns([1, 4])
                with col1:
                    allow_internet = st.checkbox("Allow internet search")
                with col2:
                    if st.button("Ask") and user_question:
                        # Add user question to chat history
                        st.session_state.chat_history.append({"role": "user", "content": user_question})
                        
                        with st.spinner("Searching for answer..."):
                            # Use the new RAG-based answer generation function
                            answer = st.session_state.assistant.answer_with_rag(user_question, allow_internet)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                        # Rerun to update the display with new messages
                        st.rerun()
            else:
                st.info("Upload and process documents to start asking questions")
                
        # Tab 3: Code Generation
        with tab3:
            st.header("Generate Code Based on Documents")
            
            if st.session_state.assistant.uploaded_files:
                # Show code history if available
                if st.session_state.assistant.code_history:
                    with st.expander("View Code History"):
                        for i, code_entry in enumerate(reversed(st.session_state.assistant.code_history)):
                            st.subheader(f"Code {len(st.session_state.assistant.code_history)-i}: {code_entry['timestamp']}")
                            st.write(f"Prompt: {code_entry['prompt']}")
                            
                            # Add a button to use this code as reference
                            if st.button(f"Build on this code", key=f"useCode_{code_entry['id']}"):
                                st.session_state.selected_code_id = code_entry['id']
                                st.session_state.selected_code = code_entry['code']
                
                # If a code was selected, show it
                selected_code = None
                if "selected_code" in st.session_state:
                    st.markdown("**Building on previous code:**")
                    st.code(st.session_state.selected_code, language="python")
                    selected_code = st.session_state.selected_code
                
                code_prompt = st.text_area("Describe the code you need:", height=100)
                
                if st.button("Generate Code") and code_prompt:
                    with st.spinner("Generating code..."):
                        # Get relevant context
                        relevant_docs = st.session_state.assistant.search_vector_db(code_prompt)
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        if not context:
                            st.warning("No relevant information found in your documents for this code request.")
                            use_internet = st.checkbox("Try to generate code without document context?")
                            if use_internet and st.button("Generate"):
                                code_response, _ = st.session_state.assistant.generate_code(
                                    code_prompt, 
                                    "No specific context available",
                                    selected_code
                                )
                                st.code(code_response, language="python")
                        else:
                            code_response, _ = st.session_state.assistant.generate_code(
                                code_prompt, 
                                context,
                                selected_code
                            )
                            
                            # Try to extract code blocks
                            code_blocks = []
                            lines = code_response.split('\n')
                            in_code_block = False
                            current_block = []
                            
                            for line in lines:
                                if line.strip().startswith('```'):
                                    if in_code_block:
                                        # End of code block
                                        code_blocks.append('\n'.join(current_block))
                                        current_block = []
                                    in_code_block = not in_code_block
                                elif in_code_block:
                                    # Skip the language identifier line
                                    if current_block or not line.strip() in ['python', 'java', 'javascript', 'html', 'css']:
                                        current_block.append(line)
                            
                            if code_blocks:
                                for i, block in enumerate(code_blocks):
                                    st.subheader(f"Code Block {i+1}")
                                    st.code(block, language="python")
                            else:
                                st.subheader("Generated Code")
                                st.code(code_response, language="python")
                            
                            # Clear selected code after generating
                            if "selected_code" in st.session_state:
                                del st.session_state.selected_code
                                del st.session_state.selected_code_id
            else:
                st.info("Upload and process documents to start generating code")

if __name__ == "__main__":
    main()

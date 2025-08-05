import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Insurance Query Assistant", 
    layout="centered",
    page_icon="🛡️"
)

st.title("🛡️ Insurance Policy Query Assistant")
st.markdown("Upload your insurance policy document and ask questions about it!")

# Backend URL
BACKEND_URL = "http://localhost:8000"

def display_enhanced_response(response_data):
    """Display response with enhanced formatting and confidence indicators"""
    
    # Main answer with confidence indicator
    confidence = response_data.get('confidence', 0)
    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
    
    st.markdown(f"""
    <div style='padding: 20px; border-left: 5px solid {confidence_color}; background-color: #f8f9fa; margin: 10px 0;'>
        <h4>📋 Answer</h4>
        <p style='font-size: 16px; line-height: 1.6;'>{response_data.get('answer', 'No answer available')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence and metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{confidence:.1%}", delta=None)
    with col2:
        st.metric("Sources Used", response_data.get('top_chunks_used', 0))
    with col3:
        processing_time = response_data.get('processing_time', 'N/A')
        st.metric("Response Time", f"{processing_time}s" if isinstance(processing_time, (int, float)) else processing_time)
    
    # Detailed breakdown
    with st.expander("🔍 Response Details"):
        if 'reasoning' in response_data:
            st.write("**Reasoning:**", response_data['reasoning'])
        
        if 'sources' in response_data and response_data['sources']:
            st.write("**Source Excerpts:**")
            for i, source in enumerate(response_data['sources'][:3]):
                st.write(f"**Source {i+1}:**")
                # Handle different source formats
                if isinstance(source, dict):
                    content = source.get('relevant_text', source.get('text', 'No content'))
                else:
                    content = str(source)
                st.code(content[:200] + "..." if len(content) > 200 else content)
        
        if 'limitations' in response_data:
            st.warning(f"⚠️ **Limitations:** {response_data['limitations']}")

def validate_uploaded_file(uploaded_file):
    """Validate file before processing"""
    max_size = 50 * 1024 * 1024  # 50MB
    allowed_types = ['application/pdf']
    
    if uploaded_file.size > max_size:
        return False, "File too large. Maximum size is 50MB."
    
    if uploaded_file.type not in allowed_types:
        return False, "Only PDF files are supported."
    
    return True, "File is valid"

def get_smart_suggestions(document_info):
    """Generate smart query suggestions based on document type"""
    base_suggestions = [
        "What is the coverage amount for this policy?",
        "What are the exclusions in this policy?",
        "What is the waiting period for claims?",
        "What is the premium payment frequency?",
        "What documents are required for claims?"
    ]
    
    # Add context-specific suggestions based on document analysis
    if document_info and document_info.get('total_chunks', 0) > 0:
        advanced_suggestions = [
            "What are the pre-existing disease conditions?",
            "What is the claim settlement process?",
            "What are the renewal conditions?",
            "What is the grace period for premium payment?",
            "What are the maternity benefits covered?"
        ]
        return base_suggestions + advanced_suggestions
    
    return base_suggestions

def validate_and_enhance_query(query):
    """Validate and suggest improvements for user queries"""
    issues = []
    suggestions = []
    
    # Check query length
    if len(query.strip()) < 10:
        issues.append("Query is too short")
        suggestions.append("Try to be more specific about what you want to know")
    
    # Check for insurance-related terms
    insurance_terms = ['policy', 'coverage', 'premium', 'claim', 'benefit', 'deductible']
    if not any(term in query.lower() for term in insurance_terms):
        suggestions.append("Consider adding insurance-specific terms like 'coverage', 'premium', or 'claim'")
    
    # Check for question words
    question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who']
    if not any(word in query.lower() for word in question_words):
        suggestions.append("Try starting with question words like 'What', 'How', or 'When'")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'suggestions': suggestions
    }

async def process_uploaded_file(file):
    """Process an uploaded file and return chunks"""
    try:
        # Prepare the file for upload
        files = {
            "file": (file.name, file.getvalue(), file.type)
        }
        
        # Send request to backend
        response = requests.post(
            f"{BACKEND_URL}/api/v1/documents/process",
            files=files,
            timeout=120
        )
        
        return response
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None

# Add API status indicator in sidebar
with st.sidebar:
    st.header("🔧 System Status")
    
    # Check backend status
    try:
        health_response = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ Backend: Online")
            
            # Show component status
            components = health_data.get('components', {})
            for component, status in components.items():
                icon = "✅" if status == "operational" else "⚠️" if status == "degraded" else "❌"
                st.write(f"{icon} {component.replace('_', ' ').title()}: {status}")
        else:
            st.error("❌ Backend: Error")
    except:
        st.error("❌ Backend: Offline")
        st.warning("🔧 Make sure the FastAPI server is running at http://localhost:8000")
        st.info("💡 Run: `python -m uvicorn main:app --reload` in the backend directory")
    
    # Show stats if available
    if 'health_data' in locals() and health_data.get('stats'):
        with st.expander("📊 System Stats"):
            st.json(health_data['stats'])

# Initialize session state
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'document_info' not in st.session_state:
    st.session_state.document_info = None

# Section 1: Document Upload
st.header("📄 Step 1: Upload Insurance Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file", 
    type=['pdf'],
    help="Upload your insurance policy document (PDF format only)"
)

if uploaded_file is not None:
    # Validate file first
    is_valid, validation_message = validate_uploaded_file(uploaded_file)
    
    if not is_valid:
        st.error(f"❌ {validation_message}")
    else:
        st.success(f"✅ File selected: {uploaded_file.name}")
        
        # Show enhanced file details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{uploaded_file.size:,} bytes")
        with col2:
            st.metric("File Type", uploaded_file.type)
        with col3:
            pages_estimate = uploaded_file.size // 50000  # Rough estimate
            st.metric("Est. Pages", f"~{pages_estimate}")
    
    # Process document button
    if st.button("🔄 Process Document", type="primary"):
        with st.spinner("Processing document... This may take a moment."):
            try:
                # Prepare the file for upload
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                # Send request to backend
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/documents/process",
                    files=files,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.document_processed = True
                    st.session_state.document_info = result.get('document_info', {})
                    
                    st.success("🎉 Document processed successfully!")
                    
                    # Show processing results
                    doc_info = st.session_state.document_info
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Chunks", doc_info.get('total_chunks', 0))
                    with col2:
                        st.metric("Characters", f"{doc_info.get('total_characters', 0):,}")
                    with col3:
                        st.metric("File Type", doc_info.get('file_type', 'unknown').upper())
                    
                    st.info("✨ Your document is ready! You can now ask questions about it below.")
                    
                else:
                    error_detail = response.json().get('detail', 'Unknown error') if response.headers.get('content-type') == 'application/json' else response.text
                    st.error(f"❌ Error processing document: {error_detail}")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The document might be too large or the server is busy.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to the backend server. Make sure it's running at http://localhost:8000")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# Section 2: Query Interface
st.header("💬 Step 2: Ask Questions")

if st.session_state.document_processed:
    st.success("✅ Document is ready for queries!")
    
    # Smart suggestions
    suggestions = get_smart_suggestions(st.session_state.document_info)
    
    st.subheader("💡 Suggested Questions")
    suggestion_cols = st.columns(2)
    
    for i, suggestion in enumerate(suggestions[:6]):  # Show top 6
        col = suggestion_cols[i % 2]
        if col.button(f"📝 {suggestion}", key=f"suggest_{i}"):
            st.session_state.selected_query = suggestion

    # Query input
    query = st.text_area(
        "Your Question:",
        value=st.session_state.get('selected_query', ''),
        height=100,
        help="Ask any question about your insurance policy"
    )

    # Real-time validation
    if query.strip():
        validation = validate_and_enhance_query(query)
        
        if not validation['is_valid']:
            for issue in validation['issues']:
                st.warning(f"⚠️ {issue}")
        
        if validation['suggestions']:
            with st.expander("💡 Query Improvement Suggestions"):
                for suggestion in validation['suggestions']:
                    st.info(f"💡 {suggestion}")

    # Query button
    if st.button("🔍 Get Answer", type="primary", disabled=not query.strip()):
        if query.strip():
            with st.spinner("Analyzing your question..."):
                try:
                    # Send query to backend
                    query_data = {"question": query.strip()}
                    response = requests.post(
                        f"{BACKEND_URL}/api/v1/query",
                        json=query_data,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        display_enhanced_response(result)
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Error: {error_detail}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Query timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to the backend server. Make sure it's running.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question.")

else:
    st.info("📤 Please upload and process a document first before asking questions.")

# Section 3: Additional Actions
st.header("🛠️ Additional Actions")

col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ Clear Database"):
        with st.spinner("Clearing database..."):
            try:
                response = requests.delete(f"{BACKEND_URL}/api/v1/documents/clear")
                if response.status_code == 200:
                    st.success("✅ Database cleared successfully!")
                    st.session_state.document_processed = False
                    st.session_state.document_info = None
                    st.rerun()
                else:
                    st.error("❌ Failed to clear database")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with col2:
    if st.button("🔍 Check System Health"):
        with st.spinner("Checking system health..."):
            try:
                response = requests.get(f"{BACKEND_URL}/api/v1/health")
                if response.status_code == 200:
                    health_data = response.json()
                    st.success("✅ System is healthy!")
                    
                    # Show system stats
                    with st.expander("System Statistics"):
                        st.json(health_data)
                else:
                    st.error("❌ System health check failed")
            except Exception as e:
                st.error(f"❌ Cannot reach backend: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>🛡️ Insurance Policy Query Assistant | Powered by LLM & Vector Search</p>
    <p>Backend API: <a href='http://localhost:8000/docs' target='_blank'>http://localhost:8000/docs</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
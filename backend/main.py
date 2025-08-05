"""
Main FastAPI Application - Hugging Face Implementation
LLM-Powered Insurance Document Query System
"""

import os
import logging
import asyncio
import tempfile
import shutil
import time
from functools import wraps
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
import sys

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insurance_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules with error handling
try:
    from .embeddings import embedding_system
    from .decision_logic import answer_question, batch_answer_questions
    from .utils import extract_text_from_pdf, clean_text, split_text_into_chunks
    logger.info("✅ All modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.info("Make sure all required modules are in the same directory")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="HackRX - LLM-Powered Insurance Document Query System",
    description="Intelligent document processing and query system using Hugging Face models",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class HackRXRequest(BaseModel):
    """HackRX submission request model"""
    documents: str = Field(..., description="Document URL to process")
    questions: List[str] = Field(..., description="List of questions to answer")

    @validator('documents')
    def validate_document_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Document URL must be a valid HTTP/HTTPS URL')
        return v

    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError('At least one question must be provided')
        if len(v) > 50:
            raise ValueError('Maximum 50 questions allowed per request')
        return [q.strip() for q in v if q.strip()]

class QueryInput(BaseModel):
    """Single query input model"""
    question: str = Field(..., description="Question to ask about the document")
    document_url: Optional[str] = Field(None, description="Optional document URL")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    components: Dict[str, str]
    stats: Dict[str, Any]

# ==================== AUTHENTICATION ====================

# def verify_hackrx_token(authorization: str = Header(None)):
#     """Verify HackRX bearer token"""
#     if not authorization:
#         raise HTTPException(status_code=401, detail="Authorization header missing")
    
#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Invalid authorization format")
    
#     token = authorization.replace("Bearer ", "")
#     expected_token = "920db1a1e34d4a69ef73ad8bcc1dd0dc2b23ea42eb973bc4e4d24d8b7bb2e3b8"
    
#     if token != expected_token:
#         raise HTTPException(status_code=401, detail="Invalid token")
    
#     return token

# ==================== UTILITY FUNCTIONS ====================

async def process_uploaded_file(file: UploadFile) -> Dict[str, Any]:
    """Process an uploaded file and return chunks"""
    try:
        logger.info(f"📄 Processing uploaded file: {file.filename}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            if file.filename.lower().endswith('.pdf'):
                text = await extract_text_from_pdf_file(temp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file.filename}")
            
            cleaned_text = clean_text(text)
            chunks = split_text_into_chunks(cleaned_text)
            
            logger.info(f"✅ Extracted {len(chunks)} chunks from {file.filename}")
            
            return {
                "status": "success",
                "filename": file.filename,
                "total_characters": len(cleaned_text),
                "total_chunks": len(chunks),
                "chunks": chunks
            }
            
        finally:
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"⚠️ Could not delete temp file: {e}")
                
    except Exception as e:
        logger.error(f"❌ Error processing file {file.filename}: {e}")
        return {
            "status": "error",
            "filename": file.filename,
            "error": str(e),
            "chunks": []
        }

async def extract_text_from_pdf_file(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"⚠️ Error extracting text from page {page_num}: {e}")
                    continue
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"❌ Error extracting text from PDF: {e}")
        raise

async def download_and_process_document(url: str) -> Dict[str, Any]:
    """Download and process document from URL"""
    try:
        from backend.ingestion import document_ingestion
        
        logger.info(f"📥 Downloading and processing: {url}")
        document_data = await document_ingestion.process_document(url)
        
        if document_data["processing_status"] != "success":
            raise Exception(f"Document processing failed: {document_data.get('error')}")
        
        return document_data
        
    except Exception as e:
        logger.error(f"❌ Error processing document from URL: {e}")
        raise

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "HackRX - LLM-Powered Insurance Document Query System",
        "version": "6.0.0",
        "status": "operational",
        "model_info": {
            "llm": "Hugging Face Transformers",
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_db": "FAISS"
        },
        "features": [
            "PDF document processing",
            "Semantic search with FAISS embeddings", 
            "Hugging Face LLM integration",
            "Structured JSON responses",
            "HackRX API compatibility"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/health",
            "hackrx": "/api/v1/hackrx/run",
            "query": "/api/v1/query"
        }
    }

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with detailed system status"""
    try:
        embedding_stats = embedding_system.get_stats()
        
        # Check Hugging Face model status
        try:
            from .decision_logic import hf_llm
            hf_status = "operational" if hf_llm else "degraded"
        except:
            hf_status = "error"
        
        components = {
            "embedding_system": "operational",
            "huggingface_llm": hf_status,
            "vector_database": "operational",
            "file_processing": "operational",
            "api": "operational"
        }
        
        system_healthy = all(status in ["operational", "degraded"] for status in components.values())
        
        return HealthResponse(
            status="healthy" if system_healthy else "degraded",
            message="HackRX system ready with Hugging Face models" if system_healthy else "Some components degraded",
            components=components,
            stats={
                "embedding_stats": embedding_stats,
                "api_version": "6.0.0",
                "model_type": "hugging_face",
                "gpu_available": True if os.environ.get("CUDA_VISIBLE_DEVICES") else False
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"System unhealthy: {str(e)}")

@app.post("/api/v1/hackrx/run")
async def hackrx_submission_endpoint(
    request: HackRXRequest,
    authorization: str = Header(None)
):
    """
    Main HackRX submission endpoint
    Process document and answer multiple questions
    """
    try:
        # Verify authentication
        # verify_hackrx_token(authorization)
        
        doc_url = request.documents
        questions = request.questions
        
        logger.info(f"🏆 HackRX Processing: {doc_url}")
        logger.info(f"📝 Questions count: {len(questions)}")

        # Step 1: Download and process document
        document_data = await download_and_process_document(doc_url)
        
        # Step 2: Add to vector database
        metadata = {
            "source_url": doc_url,
            "file_type": document_data.get("file_type", "unknown"),
            "total_chunks": document_data.get("total_chunks", 0),
            "hackrx_submission": True
        }

        db_name = embedding_system.add_documents(document_data["chunks"], metadata)
        logger.info(f"📥 Added {len(document_data['chunks'])} chunks to vector DB")

        # Step 3: Process all questions using batch processing
        logger.info("🔄 Processing questions with Hugging Face LLM...")
        
        question_results = batch_answer_questions(questions, db_name=db_name)
        
        # Step 4: Extract answers in HackRX format
        answers = []
        for i, result in enumerate(question_results):
            answer = result.get("answer", "Unable to process this question.")
            confidence = result.get("confidence", 0.0)
            
            # Clean up the answer
            answer = answer.strip()
            if answer.startswith("Based on"):
                answer = answer.replace("Based on the provided context, ", "")
                answer = answer.replace("Based on the insurance policy context, ", "")
            
            answers.append(answer)
            logger.info(f"Q{i+1}: {confidence:.2f} confidence")

        logger.info(f"✅ HackRX Processing complete: {len(answers)} answers generated")
        
        return {"answers": answers}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HackRX processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/v1/query")
async def single_query(
    request: QueryInput,
    authorization: str = Header(None)
):
    """Query the processed documents or process new document"""
    try:
        # verify_hackrx_token(authorization)
        
        question = request.question
        logger.info(f"🔍 Processing query: {question[:50]}...")
        
        # If document URL provided, process it first
        if request.document_url:
            document_data = await download_and_process_document(request.document_url)
            metadata = {
                "source_url": request.document_url,
                "file_type": document_data.get("file_type", "unknown"),
                "total_chunks": document_data.get("total_chunks", 0)
            }
            db_name = embedding_system.add_documents(document_data["chunks"], metadata)
        else:
            # Check if we have any documents in the database
            stats = embedding_system.get_stats()
            if stats["total_chunks"] == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No documents available. Please provide a document_url or upload a document first."
                )
            db_name = None
        
        # Get detailed answer
        result = answer_question(question, db_name)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/api/v1/documents/process")
async def process_document(
    file: UploadFile = File(...),
    # authorization: str = Header(None)
):
    """Process a document by uploading it directly"""
    try:
        # verify_hackrx_token(authorization)
        logger.info(f"🔄 Processing document: {file.filename}")
        
        file_data = await process_uploaded_file(file)
        
        if file_data["status"] != "success":
            raise HTTPException(
                status_code=400,
                detail=f"Document processing failed: {file_data.get('error', 'Unknown error')}"
            )
        
        metadata = {
            "source": "uploaded_file",
            "filename": file.filename,
            "file_type": Path(file.filename).suffix.lower().replace('.', ''),
            "total_chunks": file_data["total_chunks"]
        }
        
        db_name = embedding_system.add_documents(file_data["chunks"], metadata)
        
        return {
            "status": "success",
            "message": f"Processed and stored {file_data['total_chunks']} chunks using Hugging Face",
            "document_info": {
                **metadata,
                "db_name": db_name,
                "total_characters": file_data.get("total_characters", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.delete("/api/v1/documents/clear")
async def clear_database(authorization: str = Header(None)):
    """Clear the vector database"""
    try:
        # verify_hackrx_token(authorization)
        embedding_system.clear_index()
        
        return {
            "status": "success",
            "message": "Vector database cleared successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Clear database error: {e}")
        raise HTTPException(status_code=500, detail=f"Clear operation failed: {str(e)}")

@app.get("/api/v1/test")
async def test_endpoint():
    """Test endpoint for HackRX"""
    return {
        "status": "operational",
        "message": "HackRX Insurance Query System - Hugging Face Edition",
        "timestamp": time.time(),
        "model_info": {
            "llm": "Hugging Face Transformers",
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_db": "FAISS"
        }
    }

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("🚀 Starting HackRX LLM-Powered Insurance Document Query System")
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("temp_docs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Initialize Hugging Face model
    try:
        from .decision_logic import hf_llm
        if hf_llm:
            logger.info("✅ Hugging Face LLM initialized")
        else:
            logger.warning("⚠️ Hugging Face LLM not available")
    except Exception as e:
        logger.error(f"❌ Error checking Hugging Face LLM: {e}")
    
    logger.info("✅ Embedding system ready")
    logger.info("✅ HackRX API endpoints configured")
    logger.info("🏆 Ready for HackRX submissions!")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("🛑 Shutting down HackRX system...")
    try:
        temp_dir = Path("temp_docs")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

# ==================== ERROR HANDLERS ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "Please check the API documentation at /docs",
            "hackrx_endpoints": [
                "/api/v1/hackrx/run",
                "/api/v1/health",
                "/api/v1/query",
                "/api/v1/test"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Something went wrong. Please try again or contact support."
        }
    )

# ==================== MAIN ====================

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    Path("temp_docs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    logger.info("🌐 Starting HackRX server at http://localhost:8000")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    logger.info("🔍 Health Check: http://localhost:8000/api/v1/health")
    logger.info("🏆 HackRX Endpoint: http://localhost:8000/api/v1/hackrx/run")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
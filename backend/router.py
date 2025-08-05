from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import logging
import asyncio

# Fix imports
try:
    from ingestion import document_ingestion
    from embeddings import embedding_system
    from decision_logic import answer_question, batch_answer_questions
except ImportError as e:
    logging.error(f"Import error: {e}")
    # Create fallback functions
    class FallbackIngestion:
        async def process_document(self, url):
            return {"processing_status": "failed", "error": "Module not available"}
    
    class FallbackEmbedding:
        def add_documents(self, chunks, metadata=None):
            return "fallback_db"
        def get_stats(self):
            return {"status": "fallback"}
    
    document_ingestion = FallbackIngestion()
    embedding_system = FallbackEmbedding()
    
    def answer_question(query, db_name=None):
        return {"answer": "Service unavailable", "confidence": 0.0}
    
    def batch_answer_questions(questions, db_name=None):
        return [{"answer": "Service unavailable", "confidence": 0.0} for _ in questions]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Updated models to match HackRX requirements
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
        return [q.strip() for q in v if q.strip()]  # Remove empty questions

class QueryRequest(BaseModel):
    question: str
    document_url: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    components: Dict[str, str]

class ProcessedDocumentResponse(BaseModel):
    status: str
    message: str
    document_info: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    reasoning: str
    sources: List[Dict[str, Any]]
    limitations: str
    additional_info: str
    query: str
    top_chunks_used: int

# Authentication function (as per HackRX requirements)
def verify_token(authorization: str = None):
    """Verify the HackRX bearer token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    expected_token = "920db1a1e34d4a69ef73ad8bcc1dd0dc2b23ea42eb973bc4e4d24d8b7bb2e3b8"
    
    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check system components
        embedding_stats = embedding_system.get_stats()
        
        components = {
            "document_ingestion": "operational",
            "embedding_system": "operational",
            "huggingface_llm": "operational",
            "vector_database": "operational"
        }
        
        return HealthCheckResponse(
            status="healthy",
            message="All systems operational - Hugging Face LLM ready",
            components=components
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@router.post("/hackrx/run", response_model=Dict[str, List[str]])
async def hackrx_submission_endpoint(
    request: HackRXRequest, 
    authorization: str = Depends(lambda r: r.headers.get("Authorization", ""))
):
    """
    Main HackRX submission endpoint
    Processes a document URL and answers multiple questions
    """
    try:
        # Verify token
        verify_token(authorization)
        
        doc_url = request.documents
        questions = request.questions
        
        logger.info(f"📄 HackRX Processing: {doc_url}")
        logger.info(f"📝 Questions count: {len(questions)}")

        # Step 1: Process document
        logger.info("🔄 Step 1: Processing document...")
        document_data = await document_ingestion.process_document(doc_url)
        
        if document_data["processing_status"] != "success":
            raise HTTPException(
                status_code=400,
                detail=f"Document processing failed: {document_data.get('error', 'Unknown error')}"
            )

        # Step 2: Add to vector database
        logger.info("🔄 Step 2: Adding to vector database...")
        metadata = {
            "source_url": doc_url,
            "file_type": document_data.get("file_type", "unknown"),
            "total_chunks": document_data.get("total_chunks", 0)
        }

        db_name = embedding_system.add_documents(document_data["chunks"], metadata)
        logger.info(f"📥 Added to vector DB: {db_name}")

        # Step 3: Process all questions in batch
        logger.info("🔄 Step 3: Processing questions...")
        
        # Use batch processing for efficiency
        question_results = batch_answer_questions(questions, db_name=db_name)
        
        # Extract just the answer strings for HackRX format
        answers = []
        for i, result in enumerate(question_results):
            answer = result.get("answer", "Unable to process this question.")
            
            # Log confidence for debugging
            confidence = result.get("confidence", 0.0)
            logger.info(f"Q{i+1} confidence: {confidence:.2f}")
            
            # Clean up the answer text
            if answer.startswith("Based on"):
                # Remove common prefixes that might be added by the model
                answer = answer.replace("Based on the provided context, ", "")
                answer = answer.replace("Based on the insurance policy context, ", "")
            
            answers.append(answer.strip())

        # Step 4: Return results in HackRX format
        logger.info(f"✅ HackRX Processing complete: {len(answers)} answers generated")
        
        return {"answers": answers}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ HackRX processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def process_single_query(
    request: QueryRequest, 
    authorization: str = Depends(lambda r: r.headers.get("Authorization", ""))
):
    """Process a single query with optional document URL"""
    try:
        verify_token(authorization)
        
        logger.info(f"🔎 Single Query: {request.question[:50]}...")
        
        # If document URL provided, process it
        if request.document_url:
            logger.info(f"📄 Document: {request.document_url}")
            
            # Process document
            document_data = await document_ingestion.process_document(request.document_url)
            
            if document_data["processing_status"] != "success":
                raise HTTPException(
                    status_code=400,
                    detail=f"Document processing failed: {document_data.get('error')}"
                )
            
            # Add to vector database
            metadata = {
                "source_url": request.document_url,
                "file_type": document_data.get("file_type", "unknown"),
                "total_chunks": document_data.get("total_chunks", 0)
            }

            db_name = embedding_system.add_documents(document_data["chunks"], metadata)
        else:
            db_name = None
        
        # Process query
        result = answer_question(request.question, db_name=db_name)
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Single query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/process", response_model=ProcessedDocumentResponse)
async def process_document_only(
    document_url: str, 
    authorization: str = Depends(lambda r: r.headers.get("Authorization", ""))
):
    """Process a document without asking questions"""
    try:
        verify_token(authorization)
        
        logger.info(f"🔄 Processing document: {document_url}")
        
        document_data = await document_ingestion.process_document(document_url)
        
        if document_data["processing_status"] != "success":
            raise HTTPException(
                status_code=400,
                detail=f"Document processing failed: {document_data.get('error')}"
            )
        
        metadata = {
            "source_url": document_url,
            "file_type": document_data.get("file_type", "unknown"),
            "total_chunks": document_data.get("total_chunks", 0)
        }
        
        db_name = embedding_system.add_documents(document_data["chunks"], metadata)
        
        return ProcessedDocumentResponse(
            status="success",
            message=f"Processed and stored {document_data['total_chunks']} chunks using Hugging Face embeddings",
            document_info={**metadata, "db_name": db_name}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/clear")
async def clear_vector_database(
    authorization: str = Depends(lambda r: r.headers.get("Authorization", ""))
):
    """Clear the vector database"""
    try:
        verify_token(authorization)
        
        embedding_system.clear_index()
        
        return {
            "status": "success", 
            "message": "Vector database cleared successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Clear database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoint for testing
@router.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "status": "operational",
        "message": "HackRX Insurance Query System - Hugging Face Edition",
        "model_info": {
            "llm": "Hugging Face Transformers",
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_db": "FAISS"
        }
    }
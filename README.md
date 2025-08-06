# LLM-Powered Insurance Document Query System

An intelligent system for processing insurance documents and answering queries using LLM and vector search. This system follows the problem statement requirements for accuracy, token efficiency, latency, reusability, and explainability.

## 🏗️ System Architecture

The system implements the following workflow as specified in the problem statement:

```
1. Input Documents (PDF Blob URL)
   ↓
2. LLM Parser (Extract structured query)
   ↓
3. Embedding Search (FAISS/Pinecone retrieval)
   ↓
4. Clause Matching (Semantic similarity)
   ↓
5. Logic Evaluation (Decision processing)
   ↓
6. JSON Output (Structured response)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (for LLM features)
- Pinecone API key (optional, for cloud vector database)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd insurance_agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here  # Optional
   PINECONE_ENVIRONMENT=us-west1-gcp  # Optional
   ```

4. **Start the system**
   ```bash
   python startup.py
   ```

   Or manually:
   ```bash
   cd backend
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Test the system**
   ```bash
   python test_system.py
   ```

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
```
Authorization: Bearer 920db1a1e34d4a69ef73ad8bcc1dd0dc2b23ea42eb973bc4e4d24d8b7bb2e3b8
```

### Main Endpoint

#### POST `/hackrx/run`
Process documents and answer questions (matches problem statement format exactly).

**Request:**
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months."
    ]
}
```

### Additional Endpoints

#### GET `/health`
Health check endpoint.

#### POST `/query`
Process a single query with detailed response.

#### POST `/documents/process`
Process a document and add to vector database.

#### DELETE `/documents/clear`
Clear the vector database.

## 🔧 System Components

### 1. Document Ingestion (`backend/ingestion.py`)
- Downloads documents from URLs
- Extracts text from PDF and DOCX files
- Cleans and preprocesses text
- Splits text into overlapping chunks

### 2. Embedding System (`backend/embeddings.py`)
- Creates semantic embeddings using Sentence Transformers
- Supports both FAISS (local) and Pinecone (cloud) vector databases
- Performs similarity search for relevant document chunks

### 3. Decision Logic (`backend/decision_logic.py`)
- Integrates with OpenAI GPT-4 for query understanding
- Extracts query intent and structure
- Analyzes relevant document chunks
- Generates explainable responses with confidence scores

### 4. API Router (`backend/router.py`)
- Handles HTTP requests and responses
- Validates input data
- Implements authentication
- Provides structured JSON responses

## 📊 Evaluation Parameters

The system is designed to meet the following evaluation criteria:

### 1. Accuracy
- **Precision of query understanding**: Uses LLM to extract structured query intent
- **Clause matching**: Semantic search finds relevant document sections
- **Confidence scoring**: Each response includes confidence level

### 2. Token Efficiency
- **Optimized prompts**: Minimal token usage in LLM calls
- **Chunk limiting**: Limits context to top 3 most relevant chunks
- **Fallback responses**: Works without LLM for basic functionality

### 3. Latency
- **Fast embedding search**: FAISS provides sub-second search
- **Async processing**: Non-blocking document processing
- **Caching**: Vector database stores processed documents

### 4. Reusability
- **Modular design**: Separate components for different functions
- **Extensible architecture**: Easy to add new document types or LLM providers
- **Configuration**: Environment-based settings

### 5. Explainability
- **Structured responses**: Include reasoning and sources
- **Query intent analysis**: Shows understanding of user intent
- **Source tracking**: Links answers to specific document chunks

## 🧪 Testing

### Comprehensive Test Suite
Run the full validation suite:
```bash
python test_system.py
```

This tests:
- ✅ System health and connectivity
- ✅ Document processing accuracy
- ✅ Query understanding and response quality
- ✅ Performance and latency
- ✅ Explainability features

### Manual Testing
1. **Health Check**: `GET http://localhost:8000/api/v1/health`
2. **Document Processing**: `POST http://localhost:8000/api/v1/documents/process`
3. **Single Query**: `POST http://localhost:8000/api/v1/query`
4. **Main Endpoint**: `POST http://localhost:8000/api/v1/hackrx/run`

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API Key Missing**
   - Create `.env` file with your API key
   - System will work with limited functionality

3. **Server Not Starting**
   - Check if port 8000 is available
   - Ensure all dependencies are installed

4. **Document Processing Errors**
   - Verify document URL is accessible
   - Check file format (PDF/DOCX supported)

### Logs
- System logs are saved to `insurance_system.log`
- Validation reports are saved to `validation_report.json`

## 🚀 Deployment

### Local Development
```bash
python startup.py
```

### Production
1. Set environment variables
2. Install dependencies
3. Run with production server:
   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Performance Metrics

The system provides comprehensive metrics:

- **Response Time**: Average query processing time
- **Accuracy**: Answer quality and relevance scores
- **Token Usage**: LLM cost optimization
- **Explainability**: Reasoning and source tracking

## 🔄 API Examples

### Example 1: Process Insurance Policy
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer 920db1a1e34d4a69ef73ad8bcc1dd0dc2b23ea42eb973bc4e4d24d8b7bb2e3b8" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the waiting period for pre-existing diseases?",
      "Does this policy cover maternity expenses?"
    ]
  }'
```

### Example 2: Single Query
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer 920db1a1e34d4a69ef73ad8bcc1dd0dc2b23ea42eb973bc4e4d24d8b7bb2e3b8" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the grace period for premium payment?",
    "document_url": "https://example.com/policy.pdf"
  }'
```

 

"""
Embeddings Module - Simplified and Fixed
Handles text embeddings using either Ollama or HuggingFace models
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import faiss
import logging
from dotenv import load_dotenv

# LangChain imports
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"  # Default to HuggingFace

# Initialize embedding model
if USE_OLLAMA:
    try:
        embed_model = OllamaEmbeddings(model="nomic-embed-text")
        logger.info("✅ Using Ollama embeddings: nomic-embed-text")
    except Exception as e:
        logger.warning(f"⚠️ Ollama failed, falling back to HuggingFace: {e}")
        USE_OLLAMA = False

if not USE_OLLAMA:
    embed_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("✅ Using HuggingFace embeddings: all-MiniLM-L6-v2")

class SimpleEmbeddingSystem:
    """Simplified embedding system using FAISS for local storage"""
    
    def __init__(self):
        """Initialize the embedding system"""
        self.embed_model = embed_model
        self.index = None
        self.chunks = []
        self.metadata_list = []
        self.dimension = None
        
        logger.info("🔄 Initializing embedding system...")
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            # Get embedding dimension by creating a test embedding
            test_embedding = self.embed_model.embed_query("test")
            self.dimension = len(test_embedding)
            
            # Create FAISS index (using cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
            
            logger.info(f"✅ FAISS index initialized with dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing FAISS index: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        try:
            if not texts:
                raise ValueError("No texts provided for embedding")
            
            logger.info(f"🔄 Creating embeddings for {len(texts)} texts")
            
            # Create embeddings using the model
            embeddings = []
            for text in texts:
                embedding = self.embed_model.embed_query(text)
                embeddings.append(embedding)
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
            
            logger.info(f"✅ Created embeddings with shape: {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {e}")
            raise
    
    def add_documents(self, chunks: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add document chunks to the vector database"""
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            
            logger.info(f"🔄 Adding {len(chunks)} chunks to vector database")
            
            # Create embeddings
            embeddings = self.create_embeddings(chunks)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store chunks and metadata
            start_idx = len(self.chunks)
            self.chunks.extend(chunks)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "chunk_index": start_idx + i,
                    "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    **(metadata or {})
                }
                self.metadata_list.append(chunk_metadata)
            
            result_msg = f"Added {len(chunks)} chunks to FAISS. Total chunks: {len(self.chunks)}"
            logger.info(f"✅ {result_msg}")
            
            return f"faiss_db_{len(self.chunks)}"  # Return a database identifier
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if not query.strip():
                raise ValueError("Empty query provided")
            
            if len(self.chunks) == 0:
                logger.warning("⚠️ No documents in index")
                return []
            
            logger.info(f"🔍 Searching for: '{query[:50]}...' (top_k={top_k})")
            
            # Create query embedding
            query_embeddings = self.create_embeddings([query])
            
            # Search in FAISS
            actual_k = min(top_k, len(self.chunks))
            scores, indices = self.index.search(query_embeddings, actual_k)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and idx != -1:  # Valid index
                    metadata = self.metadata_list[idx] if idx < len(self.metadata_list) else {}
                    results.append({
                        "text": self.chunks[idx],
                        "score": float(score),
                        "metadata": metadata,
                        "chunk_index": idx
                    })
            
            logger.info(f"✅ Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching: {e}")
            raise
    
    def get_relevant_chunks(self, query: str, db_name: str = None, top_k: int = 3) -> List[str]:
        """Get relevant chunks for a query (compatibility method for decision_logic.py)"""
        try:
            results = self.search(query, top_k=top_k)
            return [result["text"] for result in results]
        except Exception as e:
            logger.error(f"❌ Error getting relevant chunks: {e}")
            return []
    
    def clear_index(self):
        """Clear the vector index"""
        try:
            logger.info("🔄 Clearing vector index...")
            
            # Reset FAISS index
            if self.dimension:
                self.index = faiss.IndexFlatIP(self.dimension)
            
            # Clear stored data
            self.chunks = []
            self.metadata_list = []
            
            logger.info("✅ Vector index cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding system"""
        return {
            "total_chunks": len(self.chunks),
            "index_dimension": self.dimension,
            "index_size": self.index.ntotal if self.index else 0,
            "model_type": "Ollama" if USE_OLLAMA else "HuggingFace"
        }

# Global instance
embedding_system = SimpleEmbeddingSystem()
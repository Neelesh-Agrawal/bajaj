"""
Decision Logic Module - Hugging Face Implementation
Handles LLM-powered query processing using Hugging Face models
"""

import os
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)

# Import our embedding system
from backend.embeddings import embedding_system

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.getenv("HF_MODEL_NAME", "microsoft/DialoGPT-medium")  # Default model
USE_GPU = torch.cuda.is_available()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Optional for some models

class HuggingFaceLLM:
    """Hugging Face LLM wrapper for insurance document analysis"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.max_length = 512
        self.device = "cuda" if USE_GPU else "cpu"
        
        logger.info(f"🤗 Initializing Hugging Face model: {self.model_name}")
        logger.info(f"📱 Device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for large models (optional)
            if USE_GPU and "7b" in self.model_name.lower():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    token=HF_TOKEN,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                # Load without quantization for smaller models
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=HF_TOKEN,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if USE_GPU else torch.float32
                )
            
            if USE_GPU:
                self.model = self.model.to(self.device)
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("✅ Hugging Face model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading Hugging Face model: {e}")
            # Fallback to a lighter model
            logger.info("🔄 Falling back to DistilGPT-2...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the primary model fails"""
        try:
            fallback_model = "distilgpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if USE_GPU:
                self.model = self.model.to(self.device)
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("✅ Fallback model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Fallback model also failed: {e}")
            raise Exception("Could not load any Hugging Face model")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using the loaded model"""
        try:
            # Truncate prompt if too long
            max_prompt_length = self.max_length - 100  # Leave room for generation
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
            
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the original prompt from the response
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            return response if response else "I couldn't generate a proper response."
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return f"Error generating response: {str(e)}"

# Initialize the Hugging Face LLM
try:
    hf_llm = HuggingFaceLLM()
    logger.info("✅ Hugging Face LLM initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Hugging Face LLM: {e}")
    hf_llm = None

# Enhanced prompt template for insurance policy analysis
def create_insurance_prompt(context: str, question: str) -> str:
    """Create a structured prompt for insurance document analysis"""
    prompt = f"""You are an expert insurance policy analyst. Based on the provided insurance policy context, answer the user's question accurately and concisely.

INSURANCE POLICY CONTEXT:
{context[:1500]}  # Limit context to avoid token overflow

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided policy context
- Be specific about coverage amounts, waiting periods, exclusions
- If information is not in the context, clearly state that
- Provide exact quotes from the policy when relevant
- Keep the answer concise but comprehensive

ANSWER:"""
    
    return prompt

def get_relevant_context(question: str, db_name: str = None, top_k: int = 3) -> tuple:
    """
    Get relevant context for a question from the vector database
    
    Args:
        question: User's question
        db_name: Database name (not used in current implementation)
        top_k: Number of relevant chunks to retrieve
    
    Returns:
        Tuple of (context_text, source_chunks)
    """
    try:
        logger.info(f"🔍 Retrieving relevant context for: {question[:50]}...")
        
        # Search for relevant chunks
        search_results = embedding_system.search(question, top_k=top_k)
        
        if not search_results:
            logger.warning("⚠️ No relevant context found")
            return "", []
        
        # Combine the top chunks
        context_chunks = []
        context_text = ""
        
        for i, result in enumerate(search_results):
            chunk_text = result["text"]
            score = result["score"]
            
            context_text += f"\n--- Policy Section {i+1} (Relevance: {score:.2f}) ---\n"
            context_text += chunk_text + "\n"
            
            context_chunks.append({
                "relevant_text": chunk_text[:300] + ("..." if len(chunk_text) > 300 else ""),
                "relevance_score": float(score),
                "full_text": chunk_text
            })
        
        logger.info(f"✅ Retrieved {len(context_chunks)} relevant chunks")
        return context_text, context_chunks
        
    except Exception as e:
        logger.error(f"❌ Error retrieving context: {e}")
        return "", []

def parse_response_to_json(raw_response: str, context_chunks: List[Dict], question: str) -> Dict[str, Any]:
    """
    Parse the raw response into structured JSON format
    """
    try:
        # Calculate confidence based on response quality
        confidence = calculate_confidence(raw_response, context_chunks)
        
        # Extract reasoning from response
        reasoning = f"Analyzed {len(context_chunks)} relevant policy sections to answer the question about insurance coverage."
        
        # Prepare sources
        sources = []
        for chunk in context_chunks[:3]:  # Limit to top 3 sources
            sources.append({
                "relevant_text": chunk.get("relevant_text", ""),
                "relevance_score": chunk.get("relevance_score", 0.0)
            })
        
        # Determine limitations
        limitations = ""
        if confidence < 0.5:
            limitations = "Low confidence due to limited relevant information in the policy document."
        elif not context_chunks:
            limitations = "No directly relevant policy sections found for this question."
        
        return {
            "answer": raw_response,
            "confidence": confidence,
            "reasoning": reasoning,
            "sources": sources,
            "limitations": limitations,
            "additional_info": f"Analysis based on {len(context_chunks)} policy sections.",
            "query": question,
            "top_chunks_used": len(context_chunks)
        }
        
    except Exception as e:
        logger.error(f"❌ Error parsing response: {e}")
        return {
            "answer": raw_response,
            "confidence": 0.5,
            "reasoning": "Basic response formatting applied",
            "sources": [],
            "limitations": "Could not perform advanced response analysis",
            "additional_info": "",
            "query": question,
            "top_chunks_used": 0
        }

def calculate_confidence(response: str, context_chunks: List[Dict]) -> float:
    """
    Calculate confidence score based on response and context quality
    """
    confidence = 0.5  # Base confidence
    
    # Factor 1: Response length and completeness
    if len(response) > 50:
        confidence += 0.1
    if len(response) > 100:
        confidence += 0.1
    
    # Factor 2: Number of relevant sources
    if len(context_chunks) >= 3:
        confidence += 0.2
    elif len(context_chunks) >= 1:
        confidence += 0.1
    
    # Factor 3: Source relevance scores
    if context_chunks:
        avg_relevance = sum(chunk.get("relevance_score", 0) for chunk in context_chunks) / len(context_chunks)
        confidence += min(avg_relevance, 0.2)
    
    # Factor 4: Insurance-specific terms in response
    insurance_terms = ["policy", "coverage", "premium", "claim", "benefit", "waiting period", "exclusion"]
    terms_found = sum(1 for term in insurance_terms if term.lower() in response.lower())
    confidence += min(terms_found * 0.05, 0.15)
    
    return min(confidence, 1.0)

def create_error_response(error_message: str, question: str) -> Dict[str, Any]:
    """Create a standardized error response"""
    return {
        "answer": f"I encountered an error while processing your question: {error_message}",
        "confidence": 0.0,
        "reasoning": "Error occurred during processing",
        "sources": [],
        "limitations": "Unable to process due to system error",
        "additional_info": "",
        "query": question,
        "top_chunks_used": 0,
        "error": error_message
    }

def answer_question(question: str, db_name: str = None) -> Dict[str, Any]:
    """Main function to answer questions using Hugging Face LLM"""
    try:
        logger.info(f"🔄 Processing question: {question[:50]}...")
        
        # Check if HF LLM is available
        if not hf_llm:
            return create_error_response("Hugging Face model not available", question)
        
        # Get relevant context
        context, sources = get_relevant_context(question, db_name, top_k=5)
        
        if not context.strip():
            return {
                "answer": "No relevant information found in the uploaded document to answer your question. Please make sure the document contains information related to your query.",
                "confidence": 0.0,
                "reasoning": "No relevant context chunks found in the vector database",
                "sources": [],
                "limitations": "Limited by available document content",
                "additional_info": "Try rephrasing your question or upload a more comprehensive document",
                "query": question,
                "top_chunks_used": 0
            }
        
        # Create prompt for Hugging Face model
        prompt = create_insurance_prompt(context, question)
        
        # Generate response using Hugging Face model
        raw_response = hf_llm.generate_response(prompt)
        
        # Parse response into structured format
        structured_response = parse_response_to_json(raw_response, sources, question)
        
        logger.info(f"✅ Question answered successfully with confidence: {structured_response['confidence']}")
        return structured_response
        
    except Exception as e:
        logger.error(f"❌ Error in answer_question: {e}")
        return create_error_response(str(e), question)

def batch_answer_questions(questions: List[str], db_name: str = None) -> List[Dict[str, Any]]:
    """
    Answer multiple questions in batch
    
    Args:
        questions: List of questions
        db_name: Database identifier (optional)
    
    Returns:
        List of response dictionaries
    """
    try:
        logger.info(f"📋 Processing batch of {len(questions)} questions")
        
        answers = []
        for i, question in enumerate(questions):
            logger.info(f"🔄 Processing question {i+1}/{len(questions)}")
            answer = answer_question(question, db_name)
            answers.append(answer)
        
        logger.info(f"✅ Batch processing complete: {len(answers)} answers generated")
        return answers
        
    except Exception as e:
        logger.error(f"❌ Batch processing error: {e}")
        return [create_error_response(str(e), q) for q in questions]

# Export the main function for backwards compatibility
__all__ = ["answer_question", "batch_answer_questions", "get_relevant_context"]
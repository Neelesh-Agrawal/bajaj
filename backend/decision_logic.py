"""
Enhanced Decision Logic Module - Improved Human-Readable Responses
Handles LLM-powered query processing with better response formatting and accuracy
"""

import os
import json
import logging
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer
)

# Import our embedding system
from backend.embeddings import embedding_system

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Using T5 for better text generation
# MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/flan-t5-base")
MODEL_NAME = os.getenv("HF_MODEL_NAME", "openai/gpt-oss-20b")
USE_GPU = False
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class EnhancedHuggingFaceLLM:
    """Enhanced Hugging Face LLM wrapper with improved response generation"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.max_length = 512
        self.device = "cuda" if USE_GPU else "cpu"
        
        logger.info(f"🤗 Initializing Enhanced Hugging Face model: {self.model_name}")
        logger.info(f"📱 Device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer with fallback strategy"""
        try:
            # Try T5-based models first (they're more reliable for Q&A)
            if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
                self._load_t5_model(self.model_name)
            else:
                self._load_causal_model(self.model_name)
                
        except Exception as e:
            logger.error(f"❌ Error loading primary model {self.model_name}: {e}")
            logger.info("🔄 Trying reliable fallback models...")
            
            # Try fallback models in order of reliability
            fallback_models = [
                ("google/flan-t5-small", "t5"),
                ("google/flan-t5-base", "t5"),
                ("microsoft/DialoGPT-medium", "causal"),
                ("gpt2", "causal")
            ]
            
            for fallback_model, model_type in fallback_models:
                try:
                    logger.info(f"🔄 Trying fallback model: {fallback_model}")
                    if model_type == "t5":
                        self._load_t5_model(fallback_model)
                    else:
                        self._load_causal_model(fallback_model)
                    
                    self.model_name = fallback_model
                    logger.info(f"✅ Successfully loaded {fallback_model}")
                    break
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback model {fallback_model} failed: {fallback_error}")
                    continue
            else:
                raise Exception("Could not load any Hugging Face model")
    
    def _load_t5_model(self, model_name: str):
        """Load T5-based model (most reliable for Q&A)"""
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
            legacy=False
        )
        
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if USE_GPU else torch.float32
        )
        
        if USE_GPU:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        self.model_type = "t5"
        logger.info(f"✅ T5 model {model_name} loaded successfully")
    
    def _load_causal_model(self, model_name: str):
        """Load causal language model (GPT-2, etc.)"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.float16 if USE_GPU else torch.float32
        )
        
        if USE_GPU:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        self.model_type = "causal"
        logger.info(f"✅ Causal model {model_name} loaded successfully")
    
    def generate_response(self, prompt: str, context_chunks: List[Dict] = None) -> str:
        """Generate enhanced human-readable response"""
        try:
            prompt = prompt.strip()
            if not prompt:
                return "Please provide a valid question about the insurance policy."
            
            # Try to generate with model first
            if hasattr(self, 'model_type'):
                if self.model_type == "t5":
                    response = self._generate_enhanced_t5_response(prompt, context_chunks)
                else:
                    response = self._generate_enhanced_causal_response(prompt, context_chunks)
            else:
                response = self._generate_template_response(prompt, context_chunks)
            
            # If model response is poor, use template-based approach
            if not self._is_response_quality_good(response):
                logger.info("🔄 Model response quality poor, using enhanced template")
                response = self._generate_enhanced_template_response(prompt, context_chunks)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in generate_response: {e}")
            return self._generate_enhanced_template_response(prompt, context_chunks)
    
    def _generate_enhanced_t5_response(self, prompt: str, context_chunks: List[Dict] = None) -> str:
        """Generate enhanced T5 response with better formatting"""
        try:
            # Create a more structured prompt
            context_text = self._extract_relevant_context(context_chunks)
            
            if context_text:
                formatted_prompt = f"""Based on the insurance policy information provided, answer this question clearly and completely:

Context: {context_text}

Question: {prompt}

Provide a clear, detailed answer:"""
            else:
                formatted_prompt = f"Answer this insurance policy question: {prompt}"

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=400,
                truncation=True,
                padding=True
            )
            
            if USE_GPU:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    min_length=30,
                    temperature=0.3,  # Lower temperature for more focused responses
                    do_sample=True,
                    top_p=0.85,
                    top_k=30,
                    repetition_penalty=1.3,
                    length_penalty=1.1,
                    early_stopping=True,
                    num_beams=3
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_response = self._clean_and_enhance_response(response, prompt, context_chunks)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"❌ Enhanced T5 generation failed: {e}")
            return self._generate_enhanced_template_response(prompt, context_chunks)
    
    def _generate_enhanced_causal_response(self, prompt: str, context_chunks: List[Dict] = None) -> str:
        """Generate enhanced causal model response"""
        try:
            context_text = self._extract_relevant_context(context_chunks)
            
            if context_text:
                formatted_prompt = f"""Insurance Policy Q&A:

Context: {context_text}

Question: {prompt}
Answer:"""
            else:
                formatted_prompt = f"Insurance Question: {prompt}\nAnswer:"

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=350,
                truncation=True
            )
            
            if USE_GPU:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.4,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.4,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer part
            if "Answer:" in full_response:
                response = full_response.split("Answer:")[-1].strip()
            else:
                response = full_response.replace(formatted_prompt, "").strip()
            
            cleaned_response = self._clean_and_enhance_response(response, prompt, context_chunks)
            return cleaned_response
            
        except Exception as e:
            logger.error(f"❌ Enhanced causal generation failed: {e}")
            return self._generate_enhanced_template_response(prompt, context_chunks)
    
    def _extract_relevant_context(self, context_chunks: List[Dict]) -> str:
        """Extract and format relevant context from chunks"""
        if not context_chunks:
            return ""
        
        context_parts = []
        for chunk in context_chunks[:5]:  # Use top 3 chunks
            if isinstance(chunk, dict):
                text = chunk.get('full_text', chunk.get('text', ''))
                if text and len(text.strip()) > 20:
                    # Clean and truncate context
                    clean_text = re.sub(r'\s+', ' ', text.strip())
                    if len(clean_text) > 200:
                        clean_text = clean_text[:200] + "..."
                    context_parts.append(clean_text)
        
        return " ".join(context_parts[:2])  # Combine top 2 contexts
    
    def _clean_and_enhance_response(self, response: str, original_question: str, context_chunks: List[Dict]) -> str:
        """Clean and enhance the response for better readability"""
        if not response:
            return self._generate_enhanced_template_response(original_question, context_chunks)
        
        # Remove common artifacts and clean up
        response = re.sub(r'[H\.!\?]{3,}', '', response)
        response = re.sub(r'[.!?]{3,}', '.', response)
        response = re.sub(r'\s+', ' ', response)
        
        # Remove repetitive patterns
        words = response.split()
        clean_words = []
        prev_word = ""
        
        for word in words:
            if word != prev_word and len(word) > 1 and word.isalnum():
                clean_words.append(word)
                prev_word = word
        
        response = ' '.join(clean_words)
        
        # If response is too short or poor quality, enhance it
        if len(response.strip()) < 20 or not self._is_response_quality_good(response):
            return self._generate_enhanced_template_response(original_question, context_chunks)
        
        # Ensure proper sentence structure
        if not response.endswith('.'):
            response += '.'
        
        # Capitalize first letter
        response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()
        
        return response.strip()
    
    def _is_response_quality_good(self, response: str) -> bool:
        """Check if the response quality is acceptable"""
        if not response or len(response.strip()) < 15:
            return False
        
        # Check for gibberish patterns
        if re.search(r'[.!?]{2,}', response) or len(response.split()) < 5:
            return False
        
        # Check for meaningful content
        words = response.split()
        meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
        
        return len(meaningful_words) >= 3
    
    def _generate_enhanced_template_response(self, prompt: str, context_chunks: List[Dict] = None) -> str:
        """Generate enhanced template-based response with context"""
        logger.info("🔄 Using enhanced template response generation...")
        
        prompt_lower = prompt.lower()
        
        # Extract key information from context if available
        context_info = self._extract_key_info_from_context(context_chunks)
        
        # Enhanced responses based on query type
        if any(word in prompt_lower for word in ['documents', 'document', 'papers', 'required', 'need']):
            if context_info.get('documents'):
                return f"According to the policy, the required documents include: {context_info['documents']}. Please ensure all documents are original or properly attested copies as specified in the policy terms."
            return "For insurance claims, you typically need to provide original or attested copies of relevant documents such as discharge summaries, medical bills, diagnostic reports, and any other supporting documentation as specified in your policy."
        
        elif any(word in prompt_lower for word in ['coverage', 'cover', 'benefit', 'included']):
            if context_info.get('coverage'):
                return f"Your policy provides coverage for: {context_info['coverage']}. The specific terms and conditions apply as detailed in your policy document."
            return "Your insurance policy provides coverage as specified in the policy schedule. Coverage details include the benefits, limits, and conditions outlined in your policy terms and conditions."
        
        elif any(word in prompt_lower for word in ['claim', 'claims', 'process', 'procedure']):
            if context_info.get('claims'):
                return f"The claim process involves: {context_info['claims']}. Please refer to the claims section of your policy for detailed procedures and requirements."
            return "To file a claim, you need to notify the insurance company promptly, submit required documents, and follow the claim procedure outlined in your policy. The specific requirements and timelines are detailed in your policy document."
        
        elif any(word in prompt_lower for word in ['premium', 'cost', 'payment', 'price']):
            if context_info.get('premium'):
                return f"Premium information: {context_info['premium']}. Payment terms and grace periods are specified in your policy schedule."
            return "Premium amounts, payment frequency, and terms are specified in your policy schedule. This includes details about payment methods, grace periods, and any applicable discounts or loadings."
        
        elif any(word in prompt_lower for word in ['waiting', 'period', 'wait']):
            if context_info.get('waiting'):
                return f"Waiting period details: {context_info['waiting']}. Different conditions may have different waiting periods as specified in your policy."
            return "Waiting periods, if applicable, are specified in your policy terms. These determine when coverage becomes effective for different types of conditions or treatments."
        
        elif any(word in prompt_lower for word in ['exclusion', 'exclude', 'not covered', 'limitation']):
            if context_info.get('exclusions'):
                return f"Policy exclusions include: {context_info['exclusions']}. These are conditions or situations not covered under your policy."
            return "Policy exclusions are specific conditions, treatments, or situations that are not covered under your insurance policy. These limitations are clearly outlined in your policy document."
        
        else:
            # Generic response with context if available
            if context_chunks:
                return f"Based on your policy document, I can provide information about coverage, claims, premiums, waiting periods, and exclusions. Your specific query relates to the policy terms and conditions. Please refer to the relevant sections of your policy for detailed information."
            return "I can help you understand various aspects of your insurance policy including coverage details, claim procedures, premium information, waiting periods, and policy exclusions. Please ask a more specific question about any of these areas."
    
    def _extract_key_info_from_context(self, context_chunks: List[Dict]) -> Dict[str, str]:
        """Extract key information from context chunks"""
        key_info = {}
        
        if not context_chunks:
            return key_info
        
        combined_text = ""
        for chunk in context_chunks[:3]:
            if isinstance(chunk, dict):
                text = chunk.get('full_text', chunk.get('text', ''))
                combined_text += f" {text}"
        
        combined_text = combined_text.lower()
        
        # Extract documents information
        if 'discharge' in combined_text or 'certificate' in combined_text or 'original' in combined_text:
            key_info['documents'] = "discharge summaries, certificates, original medical bills, and attested copies of relevant medical documents"
        
        # Extract coverage information
        if 'coverage' in combined_text or 'benefit' in combined_text:
            key_info['coverage'] = "the benefits and coverage amounts as specified in your policy schedule"
        
        # Extract claims information
        if 'claim' in combined_text:
            key_info['claims'] = "submitting required documents, providing necessary information, and following the prescribed claim procedures"
        
        # Extract premium information
        if 'premium' in combined_text or 'payment' in combined_text:
            key_info['premium'] = "amounts and payment terms as detailed in your policy schedule"
        
        # Extract waiting period information
        if 'waiting' in combined_text or 'period' in combined_text:
            key_info['waiting'] = "specific waiting periods for different conditions as outlined in your policy terms"
        
        # Extract exclusions information
        if 'exclusion' in combined_text or 'exclude' in combined_text:
            key_info['exclusions'] = "specific conditions and situations as detailed in the exclusions section of your policy"
        
        return key_info

# Initialize the Enhanced Hugging Face LLM
try:
    logger.info("🚀 Initializing Enhanced Hugging Face LLM...")
    hf_llm = EnhancedHuggingFaceLLM()
    logger.info("✅ Enhanced Hugging Face LLM initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Enhanced Hugging Face LLM: {e}")
    hf_llm = None

def create_enhanced_insurance_prompt(context: str, question: str) -> str:
    """Create an enhanced structured prompt for insurance document analysis"""
    # Keep context focused and relevant
    context = context[:1000] if len(context) > 1000 else context
    
    prompt = f"""You are an insurance policy expert. Answer the following question based on the provided policy information.

Policy Information: {context}

Question: {question}

Provide a clear, comprehensive answer that addresses the question directly. Include specific details from the policy when available.

Answer:"""
    
    return prompt

def get_enhanced_relevant_context(question: str, db_name: str = None, top_k: int = 4) -> tuple:
    """Get enhanced relevant context for a question from the vector database"""
    try:
        logger.info(f"🔍 Retrieving enhanced relevant context for: {question[:50]}...")
        
        # Search for relevant chunks with higher top_k for better context
        search_results = embedding_system.search(question, top_k=top_k)
        
        if not search_results:
            logger.warning("⚠️ No relevant context found")
            return "", []
        
        # Process and rank results
        context_chunks = []
        context_text = ""
        
        for i, result in enumerate(search_results):
            chunk_text = result["text"]
            score = result["score"]
            
            # Only include high-quality chunks
            if score > 0.3 and len(chunk_text.strip()) > 50:
                context_text += f"{chunk_text.strip()}\n\n"
                
                context_chunks.append({
                    "relevant_text": chunk_text[:400] + ("..." if len(chunk_text) > 400 else ""),
                    "relevance_score": float(score),
                    "full_text": chunk_text,
                    "rank": i + 1
                })
        
        # Sort by relevance score
        context_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        logger.info(f"✅ Retrieved {len(context_chunks)} high-quality relevant chunks")
        return context_text.strip(), context_chunks
        
    except Exception as e:
        logger.error(f"❌ Error retrieving enhanced context: {e}")
        return "", []

def parse_enhanced_response_to_json(raw_response: str, context_chunks: List[Dict], question: str, processing_time: float = None) -> Dict[str, Any]:
    """Parse the raw response into enhanced structured JSON format"""
    try:
        confidence = calculate_enhanced_confidence(raw_response, context_chunks, question)
        
        # Enhanced reasoning based on context quality and question complexity
        reasoning_parts = []
        reasoning_parts.append(f"Analyzed {len(context_chunks)} relevant policy sections")
        
        if context_chunks:
            avg_score = sum(chunk.get("relevance_score", 0) for chunk in context_chunks) / len(context_chunks)
            if avg_score > 0.7:
                reasoning_parts.append("with high relevance to your question")
            elif avg_score > 0.5:
                reasoning_parts.append("with good relevance to your question")
            else:
                reasoning_parts.append("with moderate relevance to your question")
        
        reasoning = " ".join(reasoning_parts) + "."
        
        # Enhanced sources with better formatting
        sources = []
        for i, chunk in enumerate(context_chunks[:3]):
            sources.append({
                "source_number": i + 1,
                "relevant_text": chunk.get("relevant_text", ""),
                "relevance_score": chunk.get("relevance_score", 0.0),
                "context_preview": chunk.get("full_text", "")[:150] + "..." if len(chunk.get("full_text", "")) > 150 else chunk.get("full_text", "")
            })
        
        # Enhanced limitations assessment
        limitations = ""
        if confidence < 0.5:
            limitations = "Lower confidence due to limited relevant information in the uploaded document."
        elif confidence < 0.7:
            limitations = "Moderate confidence. Additional policy details may provide more comprehensive information."
        elif len(context_chunks) < 2:
            limitations = "Limited context available. Response based on available policy information."
        
        return {
            "answer": raw_response,
            "confidence": confidence,
            "reasoning": reasoning,
            "sources": sources,
            "limitations": limitations,
            "additional_info": f"Response based on analysis of {len(context_chunks)} policy sections with average relevance of {sum(chunk.get('relevance_score', 0) for chunk in context_chunks) / len(context_chunks):.1%}" if context_chunks else "Response generated from available policy information",
            "query": question,
            "top_chunks_used": len(context_chunks),
            "processing_time": processing_time or "N/A",
            "response_quality": "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "basic"
        }
        
    except Exception as e:
        logger.error(f"❌ Error parsing enhanced response: {e}")
        return {
            "answer": raw_response,
            "confidence": 0.3,
            "reasoning": "Standard response formatting applied due to processing error",
            "sources": [],
            "limitations": "Technical processing limitations encountered",
            "additional_info": f"Error in response processing: {str(e)}",
            "query": question,
            "top_chunks_used": 0,
            "processing_time": processing_time or "N/A",
            "response_quality": "basic"
        }

def calculate_enhanced_confidence(response: str, context_chunks: List[Dict], question: str) -> float:
    """Calculate enhanced confidence score with multiple factors"""
    confidence = 0.2  # Base confidence
    
    # Response quality factors
    if len(response) > 50:
        confidence += 0.15
    if len(response) > 100:
        confidence += 0.1
    if len(response) > 200:
        confidence += 0.05
    
    # Context quality factors
    if len(context_chunks) >= 1:
        confidence += 0.1
    if len(context_chunks) >= 2:
        confidence += 0.1
    if len(context_chunks) >= 3:
        confidence += 0.1
    
    # Relevance factors
    if context_chunks:
        scores = [chunk.get("relevance_score", 0) for chunk in context_chunks]
        avg_relevance = sum(scores) / len(scores)
        max_relevance = max(scores)
        
        confidence += min(avg_relevance * 0.2, 0.15)
        confidence += min(max_relevance * 0.1, 0.05)
    
    # Question-answer alignment (simple heuristic)
    question_words = set(question.lower().split())
    response_words = set(response.lower().split())
    word_overlap = len(question_words.intersection(response_words))
    
    if word_overlap > 2:
        confidence += 0.05
    if word_overlap > 4:
        confidence += 0.05
    
    # Response completeness
    if any(term in response.lower() for term in ['according to', 'based on', 'policy', 'coverage']):
        confidence += 0.05
    
    return min(confidence, 1.0)

def answer_question_enhanced(question: str, db_name: str = None) -> Dict[str, Any]:
    """Enhanced main function to answer questions using improved LLM"""
    import time
    start_time = time.time()
    
    try:
        logger.info(f"🔄 Processing enhanced question: {question[:50]}...")
        
        # Check if Enhanced HF LLM is available
        if not hf_llm:
            processing_time = time.time() - start_time
            return create_enhanced_error_response("Enhanced language model not available", question, processing_time)
        
        # Get enhanced relevant context
        context, sources = get_enhanced_relevant_context(question, db_name, top_k=4)
        
        if not context.strip():
            processing_time = time.time() - start_time
            return {
                "answer": "I don't have sufficient relevant information in the uploaded document to provide a detailed answer to your question. Please ensure your document contains comprehensive insurance policy details related to your query, or try asking about specific policy aspects like coverage details, claim procedures, premium information, or policy terms.",
                "confidence": 0.0,
                "reasoning": "No relevant context found in the uploaded document",
                "sources": [],
                "limitations": "Insufficient document content or query not related to available policy information",
                "additional_info": "Consider uploading a more comprehensive policy document or asking about coverage, claims, premiums, waiting periods, or exclusions",
                "query": question,
                "top_chunks_used": 0,
                "processing_time": round(processing_time, 2),
                "response_quality": "basic"
            }
        
        # Create enhanced prompt
        prompt = create_enhanced_insurance_prompt(context, question)
        
        # Generate enhanced response
        logger.info("🤖 Generating enhanced response...")
        raw_response = hf_llm.generate_response(prompt, sources)
        
        if not raw_response or len(raw_response.strip()) < 10:
            raw_response = f"Based on the available insurance policy information, I can provide guidance on your question about {question}. The policy contains relevant information, but a more specific answer requires additional context. Please refer to the specific sections of your policy document for detailed information."
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Structure the enhanced response
        structured_response = parse_enhanced_response_to_json(raw_response, sources, question, round(processing_time, 2))
        
        logger.info(f"✅ Enhanced response generated successfully (confidence: {structured_response['confidence']:.1%}, quality: {structured_response['response_quality']})")
        return structured_response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Error in answer_question_enhanced: {e}")
        return create_enhanced_error_response(str(e), question, processing_time)

def create_enhanced_error_response(error_message: str, question: str, processing_time: float) -> Dict[str, Any]:
    """Create an enhanced standardized error response"""
    return {
        "answer": "I encountered a technical issue while processing your question. Please try rephrasing your question about the insurance policy, or check if your document contains relevant information about coverage, claims, premiums, or policy terms.",
        "confidence": 0.0,
        "reasoning": "Technical error prevented proper analysis",
        "sources": [],
        "limitations": f"System error: {error_message}",
        "additional_info": "Please ensure the backend system is properly configured and your document contains insurance policy information",
        "query": question,
        "top_chunks_used": 0,
        "processing_time": round(processing_time, 2),
        "response_quality": "error",
        "error": error_message
    }

# Export the enhanced functions
__all__ = ["answer_question_enhanced", "get_enhanced_relevant_context", "hf_llm"]

#!/usr/bin/env python3
"""
HackRX API Tester - Hugging Face Implementation
Test your Insurance Query System API with HackRX format
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
HACKRX_TOKEN = "920db1a1e34d4a69ef73ad8bcc1dd0dc2b23ea42eb973bc4e4d24d8b7bb2e3b8"
TEST_DOCUMENT = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

HEADERS = {
    "Authorization": f"Bearer {HACKRX_TOKEN}",
    "Content-Type": "application/json"
}

def test_health():
    """Test health endpoint"""
    try:
        logger.info("🔍 Testing health endpoint...")
        response = requests.get(f"{BASE_URL}/api/v1/health")
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Health check passed")
            logger.info(f"Status: {data.get('status')}")
            logger.info(f"Components: {data.get('components')}")
            return True
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return False

def test_hackrx_endpoint():
    """Test the main HackRX endpoint with Hugging Face"""
    try:
        payload = {
            "documents": TEST_DOCUMENT,
            "questions": [
                "What is the grace period for premium payment under this policy?",
                "What is the waiting period for pre-existing diseases?",
                "Does this policy cover maternity expenses?",
                "What is the waiting period for cataract surgery?",
                "Are medical expenses for organ donors covered?",
                "What is the No Claim Discount offered?",
                "Is there a benefit for preventive health check-ups?",
                "How does the policy define a Hospital?",
                "What is the coverage for AYUSH treatments?",
                "Are there sub-limits on room rent for Plan A?"
            ]
        }
        
        logger.info("🏆 Testing HackRX endpoint with Hugging Face LLM...")
        logger.info(f"Document: {TEST_DOCUMENT[:50]}...")
        logger.info(f"Questions: {len(payload['questions'])}")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/hackrx/run", 
            json=payload, 
            headers=HEADERS,
            timeout=300  # 5 minutes timeout for Hugging Face processing
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get("answers", [])
            
            logger.info(f"✅ HackRX test passed!")
            logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
            logger.info(f"📝 Received {len(answers)} answers")
            
            # Display sample answers
            for i, answer in enumerate(answers[:3]):  # Show first 3 answers
                logger.info(f"Q{i+1}: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            
            # Validate response format
            if isinstance(answers, list) and len(answers) == len(payload['questions']):
                logger.info("✅ Response format is correct")
                return True
            else:
                logger.error(f"❌ Response format error: expected {len(payload['questions'])} answers, got {len(answers)}")
                return False
                
        else:
            logger.error(f"❌ HackRX test failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("❌ HackRX test timed out (Hugging Face model might be slow)")
        return False
    except Exception as e:
        logger.error(f"❌ HackRX test error: {e}")
        return False

def test_single_query():
    """Test single query endpoint"""
    try:
        payload = {
            "question": "What is the grace period for premium payment?",
            "document_url": TEST_DOCUMENT
        }
        
        logger.info("🔍 Testing single query endpoint...")
        response = requests.post(
            f"{BASE_URL}/api/v1/query", 
            json=payload, 
            headers=HEADERS,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            confidence = data.get("confidence", 0)
            
            logger.info(f"✅ Single query test passed")
            logger.info(f"🎯 Confidence: {confidence:.2f}")
            logger.info(f"📝 Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            return True
        else:
            logger.error(f"❌ Single query test failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Single query test error: {e}")
        return False

def test_system_info():
    """Test system information endpoints"""
    try:
        logger.info("ℹ️ Testing system info...")
        
        # Test root endpoint
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ System version: {data.get('version')}")
            logger.info(f"🤗 Model info: {data.get('model_info')}")
        
        # Test dedicated test endpoint
        response = requests.get(f"{BASE_URL}/api/v1/test")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Test endpoint: {data.get('status')}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ System info test error: {e}")
        return False

def benchmark_performance():
    """Run performance benchmark"""
    try:
        logger.info("⚡ Running performance benchmark...")
        
        simple_questions = [
            "What is covered by this policy?",
            "What are the exclusions?",
            "What is the premium amount?"
        ]
        
        payload = {
            "documents": TEST_DOCUMENT,
            "questions": simple_questions
        }
        
        # Measure processing time
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/hackrx/run", 
            json=payload, 
            headers=HEADERS,
            timeout=180
        )
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get("answers", [])
            
            # Calculate metrics
            questions_per_second = len(simple_questions) / total_time
            avg_time_per_question = total_time / len(simple_questions)
            
            logger.info(f"✅ Performance benchmark completed")
            logger.info(f"⏱️ Total time: {total_time:.2f} seconds")
            logger.info(f"📊 Questions/second: {questions_per_second:.2f}")
            logger.info(f"⏳ Avg time per question: {avg_time_per_question:.2f} seconds")
            
            return True
        else:
            logger.error(f"❌ Performance benchmark failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Performance benchmark error: {e}")
        return False

def main():
    """Run comprehensive API tests"""
    logger.info("🧪 Starting HackRX API Tests - Hugging Face Edition")
    logger.info("=" * 60)
    
    # Check server availability
    logger.info("📡 Checking server availability...")
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code == 200:
            logger.info("✅ Server is running")
        else:
            logger.error("❌ Server not responding properly")
            return
    except Exception as e:
        logger.error(f"❌ Cannot connect to server: {e}")
        logger.info("Make sure the server is running: python main.py")
        return
    
    # Run
#!/usr/bin/env python3
"""
Startup script for the LLM-Powered Insurance Document Query System
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import openai
        import sentence_transformers
        import faiss
        logger.info("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install dependencies with: pip install -r requirements.txt")
        return False

def setup_environment():
    """Set up environment variables"""
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if not env_file.exists():
        if env_example.exists():
            logger.info("📝 Creating .env file from template...")
            with open(env_example, 'r') as f:
                template = f.read()
            
            with open(env_file, 'w') as f:
                f.write(template)
            
            logger.info("✅ .env file created. Please edit it with your API keys.")
            logger.warning("⚠️  You need to set OPENAI_API_KEY in .env file")
            return False
        else:
            logger.error("❌ No .env file or env_example.txt found")
            return False
    
    # Check if OpenAI API key is set
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️  OPENAI_API_KEY not set in .env file")
        logger.info("Please add your OpenAI API key to the .env file")
        return False
    
    logger.info("✅ Environment variables configured")
    return True

def create_temp_directories():
    """Create necessary temporary directories"""
    temp_dirs = ["temp_docs", "logs"]
    
    for dir_name in temp_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            logger.info(f"📁 Created directory: {dir_name}")

def start_server():
    """Start the FastAPI server"""
    try:
        logger.info("🚀 Starting LLM-Powered Insurance Document Query System...")
        
        # Change to backend directory
        backend_dir = Path("backend")
        if not backend_dir.exists():
            logger.error("❌ Backend directory not found")
            return False
        
        os.chdir(backend_dir)
        
        # Start server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ]
        
        logger.info("🌐 Server starting at http://localhost:8000")
        logger.info("📚 API Documentation: http://localhost:8000/docs")
        logger.info("🔍 Health Check: http://localhost:8000/api/v1/health")
        logger.info("⏹️  Press Ctrl+C to stop the server")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("=" * 60)
    print("LLM-Powered Insurance Document Query System")
    print("=" * 60)
    
    # Pre-flight checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment", setup_environment),
    ]
    
    for check_name, check_func in checks:
        logger.info(f"\n🔍 Running {check_name} check...")
        if not check_func():
            logger.error(f"❌ {check_name} check failed")
            return False
    
    # Create directories
    create_temp_directories()
    
    # Start server
    return start_server()

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.error("❌ python-dotenv not installed. Please run: pip install python-dotenv")
        sys.exit(1)
    
    success = main()
    if not success:
        logger.error("❌ Startup failed. Please check the errors above.")
        sys.exit(1) 
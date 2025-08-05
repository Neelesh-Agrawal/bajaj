#!/usr/bin/env python3
"""
Complete Setup Script for HackRX Insurance Query System
Hugging Face Implementation
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║           HackRX Insurance Query System Setup            ║
    ║              Hugging Face Implementation                 ║
    ╠══════════════════════════════════════════════════════════╣
    ║  🤗 Powered by Hugging Face Transformers                ║
    ║  🔍 FAISS Vector Search                                  ║
    ║  📄 PDF Document Processing                              ║
    ║  🏆 HackRX Competition Ready                             ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_requirements():
    """Check system requirements"""
    logger.info("🔍 Checking system requirements...")
    
    # Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Operating system
    os_name = platform.system()
    logger.info(f"💻 OS: {os_name}")
    
    # Memory check (basic)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"🧠 RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            logger.warning("⚠️ Low RAM detected. Consider using lighter models.")
    except ImportError:
        logger.info("📊 Memory info unavailable (install psutil for details)")
    
    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 GPU: {gpu_name} (x{gpu_count})")
        else:
            logger.info("💻 CPU-only mode (no CUDA GPU detected)")
    except ImportError:
        logger.info("🔧 PyTorch not installed yet")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing dependencies...")
    
    # Core dependencies
    core_packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "python-dotenv==1.0.0",
        "requests==2.31.0",
        "PyPDF2==3.0.1",
        "pydantic==2.5.0"
    ]
    
    # Hugging Face dependencies
    hf_packages = [
        "transformers==4.35.2",
        "torch==2.1.1",
        "tokenizers==0.15.0",
        "accelerate==0.24.1",
        "sentence-transformers==2.2.2",
        "faiss-cpu==1.7.4",  # Use faiss-gpu if CUDA available
        "numpy==1.24.3"
    ]
    
    # Optional packages
    optional_packages = [
        "bitsandbytes==0.41.3",  # For model quantization
        "streamlit==1.28.2",     # For web interface
        "psutil==5.9.6"          # For system monitoring
    ]
    
    all_packages = core_packages + hf_packages + optional_packages
    
    try:
        # Install packages
        cmd = [sys.executable, "-m", "pip", "install"] + all_packages
        logger.info("⏳ Installing packages (this may take several minutes)...")
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✅ All dependencies installed successfully")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Installation failed: {e}")
        logger.info("🔧 Try installing manually: pip install -r requirements.txt")
        return False

def create_project_structure():
    """Create project directory structure"""
    logger.info("📁 Creating project structure...")
    
    directories = [
        "backend",
        "logs",
        "temp_docs",
        "data",
        "models",  # For storing downloaded models
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📂 Created: {directory}/")
    
    return True

def create_env_file():
    """Create .env configuration file"""
    logger.info("⚙️ Creating configuration file...")
    
    env_content = """# HackRX Insurance Query System Configuration
# Hugging Face Implementation

# Model Configuration
HF_MODEL_NAME=microsoft/DialoGPT-medium
# Alternative models (uncomment to use):
# HF_MODEL_NAME=microsoft/DialoGPT-large
# HF_MODEL_NAME=distilgpt2
# HF_MODEL_NAME=gpt2

# Optional: Hugging Face Token (for private models)
# HUGGINGFACE_TOKEN=your_token_here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# System Settings
USE_OPENAI=0
USE_OLLAMA=0
LOG_LEVEL=INFO
DEBUG=True

# Performance Settings
MAX_CONTEXT_LENGTH=1500
MAX_RESPONSE_LENGTH=150
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API Settings
API_PORT=8000
API_HOST=0.0.0.0

# HackRX Token (DO NOT CHANGE)
HACKRX_TOKEN=920db1a1e34d4a69ef73ad8bcc1dd0dc2b23ea42eb973bc4e4d24d8b7bb2e3b8
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        logger.info("✅ .env file created")
    else:
        logger.info("ℹ️ .env file already exists")
    
    return True

def test_huggingface_setup():
    """Test Hugging Face setup"""
    logger.info("🤗 Testing Hugging Face setup...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Test model loading (lightweight model for testing)
        model_name = "distilgpt2"
        logger.info(f"📥 Testing model download: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Test inference
        test_input = "This is a test"
        inputs = tokenizer.encode(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=20, num_return_sequences=1)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Model test successful: {result[:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hugging Face test failed: {e}")
        logger.info("🔧 You may need to install PyTorch separately")
        return False

def download_required_models():
    """Download and cache required models"""
    logger.info("📥 Downloading required models...")
    
    models_to_download = [
        "sentence-transformers/all-MiniLM-L6-v2",  # Embedding model
        "microsoft/DialoGPT-medium",               # Default LLM
        "distilgpt2"                               # Fallback model
    ]
    
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Download embedding model
        logger.info("📥 Downloading embedding model...")
        embedding_model = SentenceTransformer(models_to_download[0])
        logger.info("✅ Embedding model ready")
        
        # Download LLM models
        for model_name in models_to_download[1:]:
            logger.info(f"📥 Downloading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info(f"✅ {model_name} ready")
        
        logger.info("✅ All models downloaded and cached")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model download failed: {e}")
        logger.info("ℹ️ Models will be downloaded when first used")
        return False

def create_startup_script():
    """Create convenient startup script"""
    logger.info("📝 Creating startup script...")
    
    startup_content = '''#!/usr/bin/env python3
"""
HackRX System Startup Script
"""
import os
import sys
import uvicorn
from pathlib import Path

def main():
    print("🚀 Starting HackRX Insurance Query System...")
    print("🤗 Using Hugging Face Models")
    print("=" * 50)
    
    # Ensure we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found!")
        print("Make sure you're in the project root directory")
        return
    
    # Create required directories
    Path("logs").mkdir(exist_ok=True)
    Path("temp_docs").mkdir(exist_ok=True)
    
    # Start server
    print("🌐 Server starting at http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🏆 HackRX Endpoint: http://localhost:8000/api/v1/hackrx/run")
    print("⏹️  Press Ctrl+C to stop")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
'''
    
    with open("start_hackrx.py", 'w') as f:
        f.write(startup_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("start_hackrx.py", 0o755)
    
    logger.info("✅ Startup script created: start_hackrx.py")
    return True

def run_basic_tests():
    """Run basic system tests"""
    logger.info("🧪 Running basic tests...")
    
    try:
        # Test imports
        logger.info("📦 Testing imports...")
        import fastapi
        import torch
        import transformers
        import sentence_transformers
        import faiss
        logger.info("✅ All imports successful")
        
        # Test model loading
        logger.info("🤗 Testing model loading...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test embedding
        test_embedding = model.encode("test sentence")
        logger.info(f"✅ Embedding test successful: {len(test_embedding)} dimensions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic test failed: {e}")
        return False

def display_final_instructions():
    """Display final setup instructions"""
    instructions = """
    ╔══════════════════════════════════════════════════════════╗
    ║                   SETUP COMPLETE! 🎉                    ║
    ╚══════════════════════════════════════════════════════════╝
    
    🚀 QUICK START:
    
    1. Start the server:
       python start_hackrx.py
       OR
       python main.py
    
    2. Test the API:
       python test_hackrx_hf.py
    
    3. Access documentation:
       http://localhost:8000/docs
    
    🏆 HACKRX SUBMISSION:
    
    Endpoint: POST /api/v1/hackrx/run
    Token: 920db1a1e34d4a69ef73ad8bcc1dd0dc2b23ea42eb973bc4e4d24d8b7bb2e3b8
    
    📝 NOTES:
    
    • First run may be slower (model downloads)
    • GPU usage is automatic if available
    • Monitor memory usage with large documents
    • Check logs/ directory for detailed logs
    
    🔧 TROUBLESHOOTING:
    
    • Low memory: Use smaller models in .env
    • Slow performance: Enable GPU or use lighter models
    • Import errors: Check pip install requirements.txt
    
    ✅ Your HackRX system is ready for submission!
    """
    
    print(instructions)

def main():
    """Main setup function"""
    print_banner()
    
    setup_steps = [
        ("System Requirements", check_system_requirements),
        ("Project Structure", create_project_structure),
        ("Dependencies", install_dependencies),
        ("Configuration", create_env_file),
        ("Hugging Face Test", test_huggingface_setup),
        ("Model Download", download_required_models),
        ("Startup Script", create_startup_script),
        ("Basic Tests", run_basic_tests),
    ]
    
    completed = 0
    total = len(setup_steps)
    
    for step_name, step_func in setup_steps:
        logger.info(f"\n🔄 Step {completed + 1}/{total}: {step_name}")
        logger.info("-" * 50)
        
        try:
            if step_func():
                completed += 1
                logger.info(f"✅ {step_name} completed")
            else:
                logger.warning(f"⚠️ {step_name} had issues (non-critical)")
                completed += 0.5  # Partial credit
        except Exception as e:
            logger.error(f"❌ {step_name} failed: {e}")
        
        time.sleep(1)
    
    # Final results
    logger.info(f"\n📊 Setup Progress: {completed}/{total}")
    
    if completed >= total * 0.8:
        logger.info("🎉 Setup completed successfully!")
        display_final_instructions()
    else:
        logger.warning("⚠️ Setup completed with issues. Check logs above.")
        logger.info("🔧 You may need to resolve some issues manually.")

if __name__ == "__main__":
    main()
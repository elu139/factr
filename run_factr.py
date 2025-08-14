#!/usr/bin/env python3
"""
factr.ai Launcher Script
Helps start the application with proper error handling and setup
"""

import os
import sys
import subprocess
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required. Current version: {}.{}".format(
            sys.version_info.major, sys.version_info.minor))
        return False
    logger.info(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'transformers', 
        'pillow', 'httpx', 'pydantic', 'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing.append(package)
            logger.warning(f"⚠️  {package} not found")
    
    if missing:
        logger.error(f"❌ Missing packages: {', '.join(missing)}")
        print(f"\n🔧 Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        print("\nOr install all requirements:")
        print("pip install -r requirements_free_tier.txt")
        return False
    
    return True

def check_redis():
    """Check if Redis is available"""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        logger.info("✅ Redis server is running")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Redis not available: {e}")
        logger.info("💡 factr.ai will run without caching (reduced performance)")
        return False

def run_startup_test():
    """Run the startup test"""
    try:
        logger.info("🧪 Running startup tests...")
        result = subprocess.run([sys.executable, "startup_test.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            logger.info("✅ Startup tests passed")
            return True
        else:
            logger.error("❌ Startup tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        logger.error(f"❌ Could not run startup tests: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    try:
        logger.info("🚀 Starting factr.ai server...")
        print("\n" + "="*60)
        print("🎯 FACTR.AI - MULTIMODAL MISINFORMATION DETECTION")
        print("="*60)
        print("📡 Server starting at: http://localhost:8000")
        print("📚 API docs: http://localhost:8000/docs")
        print("🖥️  Frontend: Open frontend_app.html in browser")
        print("⏹️  Stop server: Ctrl+C")
        print("="*60)
        
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

def main():
    """Main launcher function"""
    print("🚀 factr.ai Launcher")
    print("=" * 30)
    
    # Check system requirements
    if not check_python_version():
        return False
    
    if not check_dependencies():
        return False
    
    # Optional checks
    check_redis()
    
    # Run tests
    if not run_startup_test():
        response = input("\n⚠️  Startup tests failed. Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Start server
    start_server()
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Launcher error: {e}")
        sys.exit(1)
#!/usr/bin/env python3
"""
Quick startup test for factr.ai
Tests that all imports work and components can be initialized
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work correctly"""
    try:
        logger.info("Testing imports...")
        
        # Core imports
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, HttpUrl
        import torch
        import numpy as np
        from PIL import Image
        
        logger.info("✅ Core imports successful")
        
        # Test CLIP import
        try:
            import clip
            logger.info("✅ CLIP import successful")
        except ImportError as e:
            logger.warning(f"⚠️  CLIP import failed: {e}")
        
        # Test transformers
        try:
            from transformers import pipeline
            logger.info("✅ Transformers import successful")
        except ImportError as e:
            logger.warning(f"⚠️  Transformers import failed: {e}")
        
        # Test Redis
        try:
            import redis
            logger.info("✅ Redis import successful")
        except ImportError as e:
            logger.warning(f"⚠️  Redis import failed: {e}")
        
        # Test NLTK
        try:
            import nltk
            logger.info("✅ NLTK import successful")
        except ImportError as e:
            logger.warning(f"⚠️  NLTK import failed: {e}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_class_definitions():
    """Test that classes can be instantiated"""
    try:
        logger.info("Testing class definitions...")
        
        # Import main classes
        from main import (
            CacheManager, 
            BERTExplanationGenerator,
            ReverseImageSearchEngine,
            ImageMetadataAnalyzer,
            InstagramScraper,
            MultimodalAnalyzer
        )
        
        logger.info("✅ All class imports successful")
        
        # Test basic instantiation (without heavy model loading)
        cache_manager = CacheManager()
        logger.info("✅ CacheManager instantiated")
        
        reverse_search = ReverseImageSearchEngine()
        logger.info("✅ ReverseImageSearchEngine instantiated")
        
        metadata_analyzer = ImageMetadataAnalyzer()
        logger.info("✅ ImageMetadataAnalyzer instantiated")
        
        instagram_scraper = InstagramScraper()
        logger.info("✅ InstagramScraper instantiated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Class definition test failed: {e}")
        return False

def test_app_creation():
    """Test FastAPI app creation"""
    try:
        logger.info("Testing FastAPI app creation...")
        
        # This will test if the app can be created without running it
        from main import app
        
        # Check if app has the expected routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/health", "/analyze/instagram", "/generate-explanation"]
        
        for expected in expected_routes:
            if expected in routes:
                logger.info(f"✅ Route {expected} found")
            else:
                logger.warning(f"⚠️  Route {expected} not found")
        
        logger.info("✅ FastAPI app creation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ App creation test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting factr.ai startup tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Class Definition Test", test_class_definitions),
        ("App Creation Test", test_app_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📋 TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! factr.ai is ready to run!")
        print("\n🎯 Next steps:")
        print("1. Start Redis server (if using caching): redis-server")
        print("2. Run the API: uvicorn main:app --reload")
        print("3. Open browser: http://localhost:8000/docs")
        print("4. Test with frontend: open frontend_app.html")
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        print("\n🔧 Install missing dependencies:")
        print("pip install -r requirements_free_tier.txt")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
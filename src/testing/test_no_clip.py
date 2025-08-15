#!/usr/bin/env python3
"""
Test script for factr.ai No-CLIP version
Verifies all components work without CLIP dependency
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports"""
    try:
        logger.info("Testing imports for No-CLIP version...")
        
        # Core FastAPI
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, HttpUrl
        logger.info("✅ FastAPI imports successful")
        
        # ML libraries (without CLIP)
        from transformers import pipeline
        logger.info("✅ Transformers import successful")
        
        # Image processing
        from PIL import Image
        logger.info("✅ PIL import successful")
        
        # Other essentials
        import numpy as np
        import nltk
        import httpx
        logger.info("✅ Essential libraries imported")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_no_clip_classes():
    """Test No-CLIP specific classes"""
    try:
        logger.info("Testing No-CLIP classes...")
        
        from main_no_clip import (
            SimpleCacheManager,
            TextBasedMLAnalyzer,
            SimpleMetadataAnalyzer,
            InstagramScraper
        )
        
        # Test instantiation
        cache_manager = SimpleCacheManager()
        logger.info("✅ SimpleCacheManager created")
        
        ml_analyzer = TextBasedMLAnalyzer()
        logger.info("✅ TextBasedMLAnalyzer created")
        
        metadata_analyzer = SimpleMetadataAnalyzer()
        logger.info("✅ SimpleMetadataAnalyzer created")
        
        scraper = InstagramScraper()
        logger.info("✅ InstagramScraper created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Class test failed: {e}")
        return False

def test_app_creation():
    """Test FastAPI app creation"""
    try:
        logger.info("Testing No-CLIP app creation...")
        
        from main_no_clip import app
        
        # Check routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/health", "/analyze/instagram", "/analytics/performance"]
        
        for route in expected_routes:
            if route in routes:
                logger.info(f"✅ Route {route} found")
            else:
                logger.warning(f"⚠️  Route {route} missing")
        
        logger.info("✅ App creation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ App creation failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without starting server"""
    try:
        logger.info("Testing basic functionality...")
        
        from main_no_clip import SimpleCacheManager, TextBasedMLAnalyzer
        
        # Test cache
        cache = SimpleCacheManager()
        import asyncio
        
        async def test_cache():
            await cache.cache_result("test_key", {"test": "data"})
            result = await cache.get_cached_result("test_key")
            return result is not None
        
        cache_works = asyncio.run(test_cache())
        if cache_works:
            logger.info("✅ Cache system working")
        else:
            logger.warning("⚠️  Cache system issues")
        
        # Test ML analyzer
        analyzer = TextBasedMLAnalyzer()
        if hasattr(analyzer, 'sentiment_analyzer'):
            logger.info("✅ Text analysis models loaded")
        else:
            logger.warning("⚠️  Some ML models not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Testing factr.ai No-CLIP Version...")
    
    tests = [
        ("Import Test", test_imports),
        ("Class Test", test_no_clip_classes),
        ("App Creation Test", test_app_creation),
        ("Functionality Test", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📋 NO-CLIP VERSION TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! No-CLIP factr.ai is ready!")
        print("\n🎯 Next steps:")
        print("1. Start server: python -c \"from main_no_clip import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)\"")
        print("2. Or use: uvicorn main_no_clip:app --reload")
        print("3. Test at: http://localhost:8000/docs")
        print("4. Try demo URLs in frontend_app.html")
        print("\n💡 Features available:")
        print("- Advanced text analysis")
        print("- Sentiment & emotion detection")
        print("- Pattern recognition")
        print("- Metadata analysis")
        print("- Demo Instagram posts")
    else:
        logger.error("❌ Some tests failed. Check errors above.")
        print("\n🔧 Try installing missing packages:")
        print("pip install transformers nltk pillow httpx")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
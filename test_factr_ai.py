#!/usr/bin/env python3
"""
Test script for factr.ai - Multimodal Misinformation Detection

This script demonstrates the capabilities of our ML-powered system
Run this after starting the FastAPI server to test functionality
"""

import asyncio
import httpx
import json
from datetime import datetime

# Test Instagram URLs (you can replace these with real ones)
TEST_URLS = [
    "https://www.instagram.com/p/test1/",  # Replace with real Instagram URLs
    "https://www.instagram.com/p/test2/",
]

async def test_health_check():
    """Test if the server is running and models are loaded"""
    print("🔍 Testing health check...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Server is healthy!")
                print(f"   ML Models: {health_data['components']['ml_models']}")
                print(f"   Capabilities: {list(health_data['capabilities'].keys())}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("   Make sure you're running: uvicorn main:app --reload")
            return False

async def test_instagram_analysis(post_url: str):
    """Test Instagram post analysis"""
    print(f"\n🔍 Testing Instagram analysis for: {post_url}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Prepare request data
            request_data = {
                "post_url": post_url,
                "include_reverse_search": True,
                "include_metadata_analysis": True
            }
            
            # Make API call
            response = await client.post(
                "http://localhost:8000/analyze/instagram",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Analysis completed successfully!")
                print(f"   Misinformation Score: {result['misinformation_score']}%")
                print(f"   Confidence Level: {result['confidence_level']}")
                print(f"   Inconsistencies Found: {len(result['detected_inconsistencies'])}")
                
                # Print detailed results
                print("\n📊 Detailed Results:")
                print(f"   Explanation: {result['explanation']}")
                
                if result['detected_inconsistencies']:
                    print("   🚨 Detected Issues:")
                    for i, issue in enumerate(result['detected_inconsistencies'][:3], 1):
                        print(f"      {i}. {issue}")
                
                print("   📈 Modality Scores:")
                for modality, score in result['modality_scores'].items():
                    print(f"      {modality}: {score:.1f}%")
                
                return result
            else:
                print(f"❌ Analysis failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return None

async def test_api_documentation():
    """Test if API documentation is accessible"""
    print("\n📚 Testing API documentation...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/docs")
            if response.status_code == 200:
                print("✅ API documentation available at: http://localhost:8000/docs")
                return True
            else:
                print(f"❌ Documentation not accessible: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot access documentation: {e}")
            return False

def display_test_summary(results):
    """Display a summary of all test results"""
    print("\n" + "="*60)
    print("📋 FACTR.AI TEST SUMMARY")
    print("="*60)
    
    if not results:
        print("❌ No successful tests completed")
        return
    
    total_tests = len(results)
    print(f"Total posts analyzed: {total_tests}")
    
    if total_tests > 0:
        avg_misinformation = sum(r['misinformation_score'] for r in results) / total_tests
        high_risk_count = sum(1 for r in results if r['misinformation_score'] > 70)
        
        print(f"Average misinformation score: {avg_misinformation:.1f}%")
        print(f"High-risk posts detected: {high_risk_count}/{total_tests}")
        
        # Show distribution
        print("\nMisinformation Score Distribution:")
        for i, result in enumerate(results, 1):
            score = result['misinformation_score']
            risk_level = "🔴 HIGH" if score > 70 else "🟡 MED" if score > 40 else "🟢 LOW"
            print(f"  Post {i}: {score:.1f}% - {risk_level}")

async def run_interactive_test():
    """Run an interactive test where user can input Instagram URLs"""
    print("\n🔄 Interactive Mode")
    print("Enter Instagram post URLs to test (or 'quit' to exit):")
    
    results = []
    
    while True:
        try:
            url = input("\nInstagram URL: ").strip()
            
            if url.lower() in ['quit', 'exit', 'q']:
                break
            
            if not url:
                continue
                
            if 'instagram.com' not in url:
                print("❌ Please enter a valid Instagram URL")
                continue
            
            result = await test_instagram_analysis(url)
            if result:
                results.append(result)
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return results

async def main():
    """Main test function"""
    print("🚀 FACTR.AI - Multimodal Misinformation Detection Test Suite")
    print("=" * 60)
    
    # Test 1: Health check
    is_healthy = await test_health_check()
    if not is_healthy:
        print("\n❌ Server is not healthy. Please fix issues before continuing.")
        return
    
    # Test 2: API documentation
    await test_api_documentation()
    
    # Test 3: Choose test mode
    print("\n🎯 Choose test mode:")
    print("1. Automated test with sample URLs")
    print("2. Interactive test (enter your own URLs)")
    print("3. Skip analysis tests")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    results = []
    
    if choice == "1":
        print("\n🤖 Running automated tests...")
        for url in TEST_URLS:
            result = await test_instagram_analysis(url)
            if result:
                results.append(result)
    
    elif choice == "2":
        results = await run_interactive_test()
    
    else:
        print("⏭️  Skipping analysis tests")
    
    # Display summary
    display_test_summary(results)
    
    print(f"\n✅ Testing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🌟 factr.ai is ready for misinformation detection!")

if __name__ == "__main__":
    asyncio.run(main())
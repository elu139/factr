#!/usr/bin/env python3
"""
factr.ai - Ultra-Minimal Version for Free Hosting
Text-based misinformation detection without heavy ML libraries
Designed to fit under 4GB Docker image limit
"""

# Core imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, List, Any
import httpx
import asyncio
import os
from datetime import datetime, timedelta
import logging
import re
import json
from PIL import Image
import io
import hashlib
import time
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
cache_manager = None
ml_analyzer = None
instagram_scraper = None

# Pydantic models
class InstagramPostRequest(BaseModel):
    post_url: HttpUrl
    include_reverse_search: bool = False
    include_metadata_analysis: bool = True
    cache_results: bool = True

class MisinformationAnalysis(BaseModel):
    misinformation_score: float
    confidence_level: str
    detected_inconsistencies: List[str]
    explanation: str
    modality_scores: Dict[str, float]
    metadata_info: Optional[Dict[str, Any]] = None
    timestamp: datetime
    processing_time: Optional[float] = None
    cache_hit: Optional[bool] = None

class InstagramPost(BaseModel):
    post_id: str
    caption: str
    image_url: str
    username: str
    timestamp: datetime
    likes: Optional[int] = None
    comments_count: Optional[int] = None

# Ultra-simple cache
class SimpleCacheManager:
    def __init__(self):
        self.cache = {}
        logger.info("✅ Simple in-memory cache ready")
    
    async def get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(key)
    
    async def cache_result(self, key: str, data: Dict[str, Any], ttl: int = 3600):
        self.cache[key] = data
    
    def generate_cache_key(self, prefix: str, *args) -> str:
        key_data = "|".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"factr_ai:{prefix}:{key_hash}"

# Ultra-lightweight text analyzer
class MinimalTextAnalyzer:
    def __init__(self):
        self.setup_patterns()
        
    def setup_patterns(self):
        """Setup regex patterns for misinformation detection"""
        self.suspicious_patterns = {
            "conspiracy": [
                r'\bthey don\'?t want you to know\b',
                r'\bmainstream media\b.*\bhiding\b',
                r'\bgovernment.*cover.*up\b',
                r'\bbig pharma\b',
                r'\bwake up.*sheep\b'
            ],
            "urgency": [
                r'\bshare before.*delet\w+\b',
                r'\bbreaking\b',
                r'\burgent\b',
                r'\balert\b'
            ],
            "emotional": [
                r'\bshocking\b',
                r'\bunbelievable\b',
                r'\bdevastating\b',
                r'\boutrageous\b',
                r'\bterrifying\b'
            ],
            "claims": [
                r'\bexclusive\b',
                r'\bleaked\b',
                r'\bsecret\b',
                r'\bhidden\b',
                r'\btruth\b.*\bexposed\b'
            ]
        }
        logger.info("✅ Pattern-based analyzer ready (no ML required)")
    
    async def analyze_cross_modal_consistency(
        self, 
        image_url: str, 
        caption: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Lightweight text analysis without heavy ML"""
        
        try:
            # Basic text analysis
            text_analysis = await self._analyze_text_patterns(caption)
            
            # Detect inconsistencies
            inconsistencies = await self._detect_pattern_inconsistencies(text_analysis, metadata)
            
            # Calculate risk score
            risk_score = await self._calculate_pattern_risk(text_analysis, inconsistencies)
            
            # Generate result
            return await self._generate_minimal_analysis(
                risk_score, inconsistencies, text_analysis, metadata
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return self._fallback_analysis()
    
    async def _analyze_text_patterns(self, caption: str) -> Dict[str, Any]:
        """Analyze text using regex patterns"""
        text = caption.lower()
        
        found_patterns = {}
        total_matches = 0
        
        for category, patterns in self.suspicious_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text):
                    matches.append(pattern)
                    total_matches += 1
            found_patterns[category] = matches
        
        # Basic metrics
        analysis = {
            "original": caption,
            "found_patterns": found_patterns,
            "total_suspicious_matches": total_matches,
            "word_count": len(caption.split()),
            "exclamation_count": caption.count('!'),
            "caps_ratio": sum(1 for c in caption if c.isupper()) / max(len(caption), 1),
            "question_count": caption.count('?')
        }
        
        return analysis
    
    async def _detect_pattern_inconsistencies(
        self, 
        text_analysis: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> List[str]:
        """Detect inconsistencies using pattern matching"""
        inconsistencies = []
        
        patterns = text_analysis["found_patterns"]
        
        # Check each category
        if patterns.get("conspiracy"):
            inconsistencies.append(f"Conspiracy language detected: {len(patterns['conspiracy'])} indicators")
        
        if patterns.get("urgency"):
            inconsistencies.append(f"Urgency manipulation detected: {len(patterns['urgency'])} indicators")
        
        if patterns.get("emotional"):
            inconsistencies.append(f"Emotional manipulation detected: {len(patterns['emotional'])} indicators")
        
        if patterns.get("claims"):
            inconsistencies.append(f"Suspicious claims detected: {len(patterns['claims'])} indicators")
        
        # Formatting issues
        if text_analysis["exclamation_count"] >= 3:
            inconsistencies.append(f"Excessive punctuation ({text_analysis['exclamation_count']} exclamations)")
        
        if text_analysis["caps_ratio"] > 0.3:
            inconsistencies.append(f"Excessive capitalization ({text_analysis['caps_ratio']:.1%})")
        
        return inconsistencies
    
    async def _calculate_pattern_risk(
        self, 
        text_analysis: Dict[str, Any], 
        inconsistencies: List[str]
    ) -> float:
        """Calculate risk score from patterns"""
        
        base_score = len(inconsistencies) * 20  # 20 points per inconsistency
        pattern_bonus = text_analysis["total_suspicious_matches"] * 10
        formatting_penalty = 0
        
        if text_analysis["exclamation_count"] >= 3:
            formatting_penalty += 15
        if text_analysis["caps_ratio"] > 0.3:
            formatting_penalty += 20
        
        total_score = base_score + pattern_bonus + formatting_penalty
        return min(100.0, total_score)
    
    async def _generate_minimal_analysis(
        self,
        risk_score: float,
        inconsistencies: List[str],
        text_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate analysis result"""
        
        # Determine confidence
        if risk_score > 70 and len(inconsistencies) >= 3:
            confidence = "High"
        elif risk_score > 40 and len(inconsistencies) >= 2:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Generate explanation
        explanation = self._generate_explanation(risk_score, inconsistencies, text_analysis)
        
        # Create modality scores
        modality_scores = {
            "pattern_analysis": risk_score,
            "conspiracy_indicators": len(text_analysis["found_patterns"].get("conspiracy", [])) * 25,
            "urgency_indicators": len(text_analysis["found_patterns"].get("urgency", [])) * 20,
            "emotional_indicators": len(text_analysis["found_patterns"].get("emotional", [])) * 15,
            "formatting_issues": min(50, text_analysis["exclamation_count"] * 10 + text_analysis["caps_ratio"] * 50)
        }
        
        return {
            "misinformation_score": round(risk_score, 1),
            "confidence_level": confidence,
            "inconsistencies": inconsistencies,
            "explanation": explanation,
            "text_analysis": text_analysis,
            "modality_scores": modality_scores,
            "analysis_mode": "minimal_pattern_matching"
        }
    
    def _generate_explanation(
        self, 
        score: float, 
        inconsistencies: List[str], 
        text_analysis: Dict[str, Any]
    ) -> str:
        """Generate explanation"""
        
        if score < 25:
            base = "Text analysis suggests low misinformation risk."
        elif score < 50:
            base = "Text shows moderate signs of misinformation patterns."
        elif score < 75:
            base = "Text shows strong misinformation indicators."
        else:
            base = "Text shows very high misinformation risk with multiple red flags."
        
        details = f" Found {text_analysis['total_suspicious_matches']} suspicious patterns."
        
        if inconsistencies:
            details += f" Key concerns: {'; '.join(inconsistencies[:2])}."
        
        return base + details + " (Lightweight analysis - upgrade for full AI detection)"
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis"""
        return {
            "misinformation_score": 50.0,
            "confidence_level": "Low",
            "inconsistencies": ["Analysis failed - technical issues"],
            "explanation": "Could not complete analysis due to technical issues.",
            "analysis_mode": "fallback"
        }

# Simple Instagram scraper with demo data
class InstagramScraper:
    def __init__(self):
        logger.info("✅ Instagram scraper ready (demo mode)")
        
    async def get_post_data(self, post_url: str) -> InstagramPost:
        """Get demo post data based on URL"""
        post_id = self._extract_post_id(post_url)
        
        # Demo captions with different risk levels
        demo_posts = [
            "BREAKING: They don't want you to know this SHOCKING truth! Government cover-up EXPOSED! Share before they delete this! Wake up sheep! #truth #exposed",
            "Beautiful sunset from my vacation in Hawaii! Had such a relaxing time with the family. #vacation #blessed #grateful",
            "URGENT ALERT! Big Pharma hiding MASSIVE secret! Mainstream media won't report this! SHARE IMMEDIATELY before censored! This will blow your mind!",
            "Just tried this amazing new recipe for dinner. Came out perfect! Thanks for the inspiration @chef_maria #cooking #homemade",
            "LEAKED: Secret government documents reveal EVERYTHING! They're hiding the truth from us! This is DEVASTATING! RT NOW!"
        ]
        
        # Select based on URL hash for consistency
        index = hash(post_url) % len(demo_posts)
        caption = demo_posts[index]
        
        return InstagramPost(
            post_id=post_id,
            caption=caption,
            image_url="https://via.placeholder.com/400x400.png?text=Demo+Image",
            username=f"demo_user_{post_id[:6]}",
            timestamp=datetime.now() - timedelta(hours=hash(post_url) % 48),
            likes=hash(post_url) % 1000 + 100,
            comments_count=hash(post_url) % 50 + 5
        )
    
    def _extract_post_id(self, post_url: str) -> str:
        """Extract post ID from URL"""
        pattern = r'/p/([A-Za-z0-9_-]+)/?'
        match = re.search(pattern, post_url)
        if match:
            return match.group(1)
        return f"demo_{abs(hash(post_url)) % 1000000}"

# Initialize components
async def initialize_components():
    """Initialize minimal components"""
    global cache_manager, ml_analyzer, instagram_scraper
    
    try:
        logger.info("🚀 Initializing factr.ai (Ultra-Minimal Mode)")
        
        cache_manager = SimpleCacheManager()
        ml_analyzer = MinimalTextAnalyzer()  
        instagram_scraper = InstagramScraper()
        
        logger.info("✅ factr.ai minimal version ready!")
        logger.info("💡 Lightweight pattern-based detection active")
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    await initialize_components()
    yield
    logger.info("🛑 Shutting down factr.ai minimal")

# FastAPI app
app = FastAPI(
    title="factr.ai - Lightweight Misinformation Detection",
    description="Pattern-based misinformation detection optimized for free hosting",
    version="1.0-minimal",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "minimal",
        "mode": "pattern_based_detection",
        "note": "Ultra-lightweight version for free hosting"
    }

@app.post("/analyze/instagram", response_model=MisinformationAnalysis)
async def analyze_instagram_post(request: InstagramPostRequest):
    """Analyze Instagram post with lightweight detection"""
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Analyzing (minimal): {request.post_url}")
        
        # Get post data
        post_data = await instagram_scraper.get_post_data(str(request.post_url))
        
        # Analyze with minimal methods
        analysis_results = await ml_analyzer.analyze_cross_modal_consistency(
            post_data.image_url,
            post_data.caption,
            {"post_timestamp": post_data.timestamp}
        )
        
        # Format response
        processing_time = time.time() - start_time
        
        response = MisinformationAnalysis(
            misinformation_score=analysis_results["misinformation_score"],
            confidence_level=analysis_results["confidence_level"],
            detected_inconsistencies=analysis_results["inconsistencies"],
            explanation=analysis_results["explanation"],
            modality_scores=analysis_results.get("modality_scores", {}),
            metadata_info={
                "post_id": post_data.post_id,
                "username": post_data.username,
                "likes": post_data.likes
            },
            timestamp=datetime.now(),
            processing_time=processing_time,
            cache_hit=False
        )
        
        logger.info(f"✅ Analysis complete: {response.misinformation_score:.1f}% risk ({processing_time:.2f}s)")
        return response
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
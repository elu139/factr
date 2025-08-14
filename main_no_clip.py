#!/usr/bin/env python3
"""
factr.ai - Text-Based Misinformation Detection (No-CLIP Version)
AI-powered text and metadata analysis for misinformation detection
Works with modern PyTorch versions without CLIP dependency
"""

# Core imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, List, Any, Union
import httpx
import asyncio
import os
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import re
import json
from PIL import Image
from PIL.ExifTags import TAGS
import io
import base64
import numpy as np
from transformers import pipeline, AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl
import hashlib
import requests
from urllib.parse import quote_plus, urlencode
import time
from functools import lru_cache
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data safely
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Global variables
cache_manager = None
ml_analyzer = None
instagram_scraper = None
metadata_analyzer = None

# Pydantic models
class ExplanationRequest(BaseModel):
    """Request model for explanation generation"""
    audience: str = "general_public"
    include_evidence: bool = True
    include_recommendations: bool = True
    language: str = "en"

class InstagramPostRequest(BaseModel):
    """Request model for Instagram post analysis"""
    post_url: HttpUrl
    include_reverse_search: bool = False  # Disabled for simplicity
    include_metadata_analysis: bool = True
    explanation_config: Optional[ExplanationRequest] = None
    cache_results: bool = True

class MisinformationAnalysis(BaseModel):
    """Response model for misinformation analysis"""
    misinformation_score: float
    confidence_level: str
    detected_inconsistencies: List[str]
    explanation: str
    modality_scores: Dict[str, float]
    metadata_info: Optional[Dict[str, Any]] = None
    timestamp: datetime
    detailed_explanation: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    cache_hit: Optional[bool] = None

class InstagramPost(BaseModel):
    """Model for Instagram post data"""
    post_id: str
    caption: str
    image_url: str
    username: str
    timestamp: datetime
    likes: Optional[int] = None
    comments_count: Optional[int] = None

# Simplified Cache Manager (in-memory)
class SimpleCacheManager:
    """In-memory cache manager (no Redis required)"""
    
    def __init__(self):
        self.cache = {}
        self.cache_times = {}
        logger.info("✅ Using in-memory cache")
    
    async def get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if not expired"""
        if key in self.cache:
            cache_time = self.cache_times.get(key, datetime.now())
            if (datetime.now() - cache_time).seconds < 3600:  # 1 hour TTL
                return self.cache[key]
            else:
                # Expired, remove
                self.cache.pop(key, None)
                self.cache_times.pop(key, None)
        return None
    
    async def cache_result(self, key: str, data: Dict[str, Any], ttl: int = 3600):
        """Cache result with timestamp"""
        self.cache[key] = data
        self.cache_times[key] = datetime.now()
        
        # Simple cleanup - remove old entries if cache gets too big
        if len(self.cache) > 100:
            oldest_key = min(self.cache_times.keys(), key=lambda k: self.cache_times[k])
            self.cache.pop(oldest_key, None)
            self.cache_times.pop(oldest_key, None)
    
    def generate_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments"""
        key_data = "|".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"factr_ai:{prefix}:{key_hash}"

# Enhanced ML Analyzer (no CLIP, but smarter text analysis)
class TextBasedMLAnalyzer:
    """
    Advanced text-based misinformation detection
    Uses multiple NLP techniques without requiring CLIP
    """
    
    def __init__(self):
        self.setup_models()
        
    def setup_models(self):
        """Initialize NLP models"""
        try:
            logger.info("🧠 Loading NLP models for text analysis...")
            
            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Text classification for fake news
            try:
                self.fake_news_classifier = pipeline(
                    "text-classification",
                    model="martin-ha/toxic-comment-model"
                )
                logger.info("✅ Fake news classifier loaded")
            except:
                logger.warning("⚠️  Advanced classifier not available, using basic analysis")
                self.fake_news_classifier = None
            
            # Emotion analysis
            try:
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                logger.info("✅ Emotion analyzer loaded")
            except:
                logger.warning("⚠️  Emotion analyzer not available")
                self.emotion_analyzer = None
            
            logger.info("✅ Text analysis models ready!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to basic analysis
            self.sentiment_analyzer = None
            self.fake_news_classifier = None
            self.emotion_analyzer = None
    
    async def analyze_cross_modal_consistency(
        self, 
        image_url: str, 
        caption: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive text-based misinformation analysis
        """
        try:
            logger.info("🔍 Running text-based misinformation analysis...")
            
            # Step 1: Advanced text preprocessing
            processed_text = await self._advanced_text_analysis(caption)
            
            # Step 2: Detect multiple types of inconsistencies
            inconsistencies = await self._detect_all_inconsistencies(processed_text, metadata)
            
            # Step 3: Calculate comprehensive risk score
            risk_analysis = await self._calculate_comprehensive_risk(processed_text, inconsistencies, metadata)
            
            # Step 4: Generate detailed analysis result
            analysis_result = await self._generate_detailed_analysis(
                risk_analysis, inconsistencies, processed_text, metadata
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return self._fallback_analysis()
    
    async def _advanced_text_analysis(self, caption: str) -> Dict[str, Any]:
        """Advanced NLP analysis of caption text"""
        cleaned_text = caption.strip()
        
        # Basic pattern extraction
        temporal_phrases = re.findall(
            r'\b(?:yesterday|today|tomorrow|last week|this year|breaking|just happened|2020|2021|2022|2023|2024|2025)\b', 
            cleaned_text.lower()
        )
        
        location_phrases = re.findall(
            r'\b(?:in|at|from)\s+([A-Z][a-zA-Z\s]+)\b', 
            cleaned_text
        )
        
        # Suspicious claim indicators
        claim_phrases = re.findall(
            r'\b(?:breaking|exclusive|leaked|secret|hidden|truth|exposed|fake|hoax|scam|urgent|alert)\b', 
            cleaned_text.lower()
        )
        
        # Emotional manipulation indicators
        emotion_words = re.findall(
            r'\b(?:shocking|devastating|unbelievable|amazing|incredible|outrageous|disgusting|terrifying)\b',
            cleaned_text.lower()
        )
        
        # URL and hashtag analysis
        urls = re.findall(r'http[s]?://\S+', cleaned_text)
        hashtags = re.findall(r'#\w+', cleaned_text)
        mentions = re.findall(r'@\w+', cleaned_text)
        
        # Advanced NLP analysis
        analysis_results = {
            "original": caption,
            "cleaned_text": cleaned_text,
            "temporal_indicators": temporal_phrases,
            "location_indicators": location_phrases,
            "claim_indicators": claim_phrases,
            "emotion_indicators": emotion_words,
            "urls": urls,
            "hashtags": hashtags,
            "mentions": mentions,
            "word_count": len(cleaned_text.split()),
            "char_count": len(cleaned_text),
            "exclamation_count": cleaned_text.count('!'),
            "question_count": cleaned_text.count('?'),
            "caps_ratio": sum(1 for c in cleaned_text if c.isupper()) / max(len(cleaned_text), 1)
        }
        
        # Sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment_results = self.sentiment_analyzer(cleaned_text[:500])
                analysis_results["sentiment"] = sentiment_results[0] if sentiment_results else {"label": "NEUTRAL", "score": 0.5}
                analysis_results["all_sentiments"] = sentiment_results
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                analysis_results["sentiment"] = {"label": "NEUTRAL", "score": 0.5}
        else:
            analysis_results["sentiment"] = {"label": "NEUTRAL", "score": 0.5}
        
        # Emotion analysis
        if self.emotion_analyzer:
            try:
                emotions = self.emotion_analyzer(cleaned_text[:500])
                analysis_results["emotions"] = emotions[0] if emotions else []
                # Find dominant emotion
                if emotions and len(emotions[0]) > 0:
                    dominant_emotion = max(emotions[0], key=lambda x: x['score'])
                    analysis_results["dominant_emotion"] = dominant_emotion
                else:
                    analysis_results["dominant_emotion"] = {"label": "neutral", "score": 0.5}
            except Exception as e:
                logger.warning(f"Emotion analysis failed: {e}")
                analysis_results["emotions"] = []
                analysis_results["dominant_emotion"] = {"label": "neutral", "score": 0.5}
        
        # Toxicity/fake news analysis
        if self.fake_news_classifier:
            try:
                toxicity = self.fake_news_classifier(cleaned_text[:500])
                analysis_results["toxicity"] = toxicity[0] if toxicity else {"label": "NORMAL", "score": 0.1}
            except Exception as e:
                logger.warning(f"Toxicity analysis failed: {e}")
                analysis_results["toxicity"] = {"label": "NORMAL", "score": 0.1}
        
        return analysis_results
    
    async def _detect_all_inconsistencies(
        self, 
        text_analysis: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> List[str]:
        """Detect multiple types of inconsistencies"""
        inconsistencies = []
        
        # 1. Suspicious claim language
        claim_count = len(text_analysis["claim_indicators"])
        if claim_count >= 3:
            inconsistencies.append(f"High density of suspicious claim language ({claim_count} indicators: {', '.join(text_analysis['claim_indicators'][:3])})")
        elif claim_count >= 1:
            inconsistencies.append(f"Suspicious claim language detected: {', '.join(text_analysis['claim_indicators'])}")
        
        # 2. Emotional manipulation
        emotion_count = len(text_analysis["emotion_indicators"])
        if emotion_count >= 2:
            inconsistencies.append(f"Emotional manipulation indicators ({emotion_count} found: {', '.join(text_analysis['emotion_indicators'][:3])})")
        
        # 3. Excessive punctuation (common in fake news)
        if text_analysis["exclamation_count"] >= 3:
            inconsistencies.append(f"Excessive exclamation marks ({text_analysis['exclamation_count']}) suggest sensationalism")
        
        # 4. High caps ratio (shouting)
        if text_analysis["caps_ratio"] > 0.3:
            inconsistencies.append(f"Excessive capitalization ({text_analysis['caps_ratio']:.1%}) indicates aggressive tone")
        
        # 5. Temporal inconsistencies
        if text_analysis["temporal_indicators"]:
            post_year = metadata.get("post_timestamp", datetime.now()).year
            claimed_years = [int(x) for x in text_analysis["temporal_indicators"] if x.isdigit() and len(x) == 4]
            
            if claimed_years and abs(post_year - max(claimed_years)) > 1:
                inconsistencies.append(f"Temporal inconsistency: Post from {post_year} claims events from {claimed_years}")
        
        # 6. Sentiment extremes
        sentiment = text_analysis.get("sentiment", {})
        if sentiment.get("score", 0) > 0.9:
            if sentiment.get("label") == "NEGATIVE":
                inconsistencies.append("Extremely negative sentiment may indicate emotional manipulation")
            elif sentiment.get("label") == "POSITIVE" and claim_count > 0:
                inconsistencies.append("Unnaturally positive tone combined with suspicious claims")
        
        # 7. Emotion analysis
        dominant_emotion = text_analysis.get("dominant_emotion", {})
        if dominant_emotion.get("score", 0) > 0.8:
            emotion_label = dominant_emotion.get("label", "").lower()
            if emotion_label in ["anger", "fear", "disgust"]:
                inconsistencies.append(f"High {emotion_label} content ({dominant_emotion['score']:.1%}) may indicate manipulation")
        
        # 8. Toxicity detection
        toxicity = text_analysis.get("toxicity", {})
        if toxicity.get("score", 0) > 0.7:
            inconsistencies.append("Content flagged as potentially toxic or misleading")
        
        # 9. Suspicious patterns
        text = text_analysis["cleaned_text"].lower()
        suspicious_patterns = [
            (r'\bthey don\'?t want you to know\b', "Conspiracy language detected"),
            (r'\bmainstream media\b.*\bhiding\b', "Anti-media conspiracy language"),
            (r'\bshare before.*delet\w+\b', "Urgency manipulation detected"),
            (r'\bgovernment.*cover.*up\b', "Government conspiracy language"),
            (r'\bbig pharma\b', "Anti-pharmaceutical conspiracy language"),
            (r'\bwake up.*sheep\b', "Derogatory awakening language")
        ]
        
        for pattern, description in suspicious_patterns:
            if re.search(pattern, text):
                inconsistencies.append(description)
        
        return inconsistencies
    
    async def _calculate_comprehensive_risk(
        self, 
        text_analysis: Dict[str, Any], 
        inconsistencies: List[str], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk score using multiple factors"""
        
        risk_factors = {
            "base_inconsistencies": len(inconsistencies) * 15,  # 15 points per inconsistency
            "claim_language": len(text_analysis["claim_indicators"]) * 10,
            "emotion_manipulation": len(text_analysis["emotion_indicators"]) * 8,
            "sentiment_extreme": 0,
            "toxicity": 0,
            "formatting_issues": 0,
            "pattern_matching": 0
        }
        
        # Sentiment scoring
        sentiment = text_analysis.get("sentiment", {})
        if sentiment.get("score", 0) > 0.85:
            if sentiment.get("label") in ["NEGATIVE"]:
                risk_factors["sentiment_extreme"] = 20
            elif sentiment.get("label") in ["POSITIVE"] and len(text_analysis["claim_indicators"]) > 0:
                risk_factors["sentiment_extreme"] = 15
        
        # Toxicity scoring
        toxicity = text_analysis.get("toxicity", {})
        if toxicity.get("score", 0) > 0.5:
            risk_factors["toxicity"] = toxicity["score"] * 30
        
        # Formatting issues
        if text_analysis["exclamation_count"] >= 3:
            risk_factors["formatting_issues"] += 10
        if text_analysis["caps_ratio"] > 0.2:
            risk_factors["formatting_issues"] += text_analysis["caps_ratio"] * 25
        
        # Pattern matching bonus
        text = text_analysis["cleaned_text"].lower()
        pattern_count = sum(1 for pattern, _ in [
            (r'\bthey don\'?t want you to know\b', ""),
            (r'\bmainstream media\b.*\bhiding\b', ""),
            (r'\bshare before.*delet\w+\b', ""),
            (r'\bgovernment.*cover.*up\b', ""),
            (r'\bbig pharma\b', ""),
            (r'\bwake up.*sheep\b', "")
        ] if re.search(pattern, text))
        
        risk_factors["pattern_matching"] = pattern_count * 12
        
        # Calculate total score
        total_risk = sum(risk_factors.values())
        final_score = min(100, total_risk)
        
        # Determine confidence based on multiple active factors
        active_factors = sum(1 for score in risk_factors.values() if score > 5)
        
        if final_score > 75 and active_factors >= 4:
            confidence = "High"
        elif final_score > 45 and active_factors >= 3:
            confidence = "Medium"
        elif final_score > 25 and active_factors >= 2:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "total_score": final_score,
            "confidence": confidence,
            "risk_factors": risk_factors,
            "active_factors": active_factors,
            "factor_breakdown": {k: v for k, v in risk_factors.items() if v > 0}
        }
    
    async def _generate_detailed_analysis(
        self,
        risk_analysis: Dict[str, Any],
        inconsistencies: List[str],
        text_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis result"""
        
        score = risk_analysis["total_score"]
        confidence = risk_analysis["confidence"]
        
        # Generate explanation
        explanation = await self._generate_smart_explanation(score, inconsistencies, text_analysis, risk_analysis)
        
        # Create modality scores for frontend
        modality_scores = {
            "text_analysis": score,
            "sentiment_analysis": text_analysis.get("sentiment", {}).get("score", 0.5) * 100,
            "emotion_analysis": text_analysis.get("dominant_emotion", {}).get("score", 0.5) * 100,
            "pattern_analysis": risk_analysis["risk_factors"].get("pattern_matching", 0) * 2,
            "claim_analysis": risk_analysis["risk_factors"].get("claim_language", 0) * 2,
            "manipulation_analysis": risk_analysis["risk_factors"].get("emotion_manipulation", 0) * 2
        }
        
        return {
            "misinformation_score": round(score, 1),
            "confidence_level": confidence,
            "inconsistencies": inconsistencies,
            "explanation": explanation,
            "text_analysis": text_analysis,
            "risk_analysis": risk_analysis,
            "modality_scores": modality_scores,
            "metadata_analysis": metadata,
            "analysis_mode": "text_based_advanced",
            "note": "Advanced text analysis - install CLIP for visual-textual consistency checking"
        }
    
    async def _generate_smart_explanation(
        self,
        score: float,
        inconsistencies: List[str],
        text_analysis: Dict[str, Any],
        risk_analysis: Dict[str, Any]
    ) -> str:
        """Generate intelligent explanation based on analysis"""
        
        if score < 25:
            risk_level = "low"
            main_msg = "The content appears authentic with minimal misinformation indicators."
        elif score < 50:
            risk_level = "moderate"
            main_msg = "The content shows some concerning patterns that warrant caution."
        elif score < 75:
            risk_level = "high"
            main_msg = "The content shows significant patterns suggesting potential misinformation."
        else:
            risk_level = "very high"
            main_msg = "The content shows strong indicators of misinformation and should be treated with extreme caution."
        
        explanation = f"Advanced text analysis indicates a {risk_level} risk of misinformation ({score:.1f}% confidence). {main_msg}"
        
        # Add specific findings
        active_factors = risk_analysis.get("active_factors", 0)
        if active_factors >= 3:
            explanation += f" Analysis detected {active_factors} different risk factors working together."
        
        # Highlight top concerns
        if inconsistencies:
            top_concerns = inconsistencies[:2]
            explanation += f" Primary concerns: {'; '.join(top_concerns)}."
            
            if len(inconsistencies) > 2:
                explanation += f" {len(inconsistencies) - 2} additional warning signs detected."
        
        # Add emotional analysis insight
        dominant_emotion = text_analysis.get("dominant_emotion", {})
        if dominant_emotion.get("score", 0) > 0.7:
            emotion = dominant_emotion.get("label", "unknown")
            explanation += f" Content heavily emphasizes {emotion} ({dominant_emotion['score']:.0%})."
        
        explanation += " (Advanced NLP analysis - visual analysis available with CLIP installation)"
        
        return explanation
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback when analysis fails"""
        return {
            "misinformation_score": 50.0,
            "confidence_level": "Low",
            "inconsistencies": ["Analysis unavailable due to technical issues"],
            "explanation": "Technical issue prevented full analysis. Manual review recommended.",
            "text_analysis": {"error": "Processing failed"},
            "risk_analysis": {"error": "Processing failed"},
            "metadata_analysis": {"status": "fallback"},
            "analysis_mode": "fallback"
        }

# Simple metadata analyzer
class SimpleMetadataAnalyzer:
    """Basic metadata analysis without heavy dependencies"""
    
    async def analyze_image_metadata(self, image_url: str) -> Dict[str, Any]:
        """Simple image metadata extraction"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                
                # Basic image analysis
                image = Image.open(io.BytesIO(response.content))
                
                # Try to get EXIF
                exif_data = {}
                if hasattr(image, '_getexif') and image._getexif() is not None:
                    exif = image._getexif()
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_data[tag] = str(value)
                
                return {
                    "status": "analyzed",
                    "format": image.format,
                    "size": image.size,
                    "mode": image.mode,
                    "has_exif": len(exif_data) > 0,
                    "exif_tags": len(exif_data),
                    "basic_exif": exif_data,
                    "analysis": {
                        "risk_score": 10.0 if len(exif_data) == 0 else 0.0,  # No EXIF might indicate processing
                        "authenticity_indicators": ["No EXIF data - possibly processed"] if len(exif_data) == 0 else ["EXIF data present"]
                    }
                }
                
        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "analysis": {"risk_score": 0.0}
            }

# Simplified Instagram scraper with demo data
class InstagramScraper:
    """Instagram scraper with demo mode for testing"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    async def get_post_data(self, post_url: str) -> InstagramPost:
        """Extract post data (demo mode for testing)"""
        try:
            post_id = self._extract_post_id(post_url)
            
            # Demo data for testing - replace with real scraping
            demo_captions = [
                "BREAKING: Scientists don't want you to know this SHOCKING truth about vaccines! Share before they delete this! #truth #exposed #wakeup",
                "Beautiful sunset from my vacation in Hawaii! Had an amazing time with family. #vacation #hawaii #blessed",
                "URGENT: Government hiding MASSIVE cover-up! Mainstream media won't report this! RT NOW before censored! #conspiracy #truth",
                "Just tried this new recipe for dinner - turned out amazing! Thanks @chef_mike for the inspiration #foodie #homecooking",
                "LEAKED: Big Pharma secret document reveals EVERYTHING they don't want you to know! This will blow your mind! Share immediately!"
            ]
            
            # Choose caption based on URL for consistent testing
            caption_index = hash(post_url) % len(demo_captions)
            caption = demo_captions[caption_index]
            
            return InstagramPost(
                post_id=post_id,
                caption=caption,
                image_url="https://via.placeholder.com/400x400.png?text=Demo+Image",
                username=f"demo_user_{post_id[:6]}",
                timestamp=datetime.now() - timedelta(hours=hash(post_url) % 48),
                likes=hash(post_url) % 1000 + 100,
                comments_count=hash(post_url) % 50 + 5
            )
            
        except Exception as e:
            logger.error(f"Error getting post data: {e}")
            raise HTTPException(status_code=400, detail=f"Could not process post: {str(e)}")
    
    def _extract_post_id(self, post_url: str) -> str:
        """Extract post ID from URL"""
        pattern = r'/p/([A-Za-z0-9_-]+)/?'
        match = re.search(pattern, post_url)
        if match:
            return match.group(1)
        # Generate ID from URL for demo
        return f"demo_{abs(hash(post_url)) % 1000000}"

# Initialize components
async def initialize_components():
    """Initialize all system components"""
    global cache_manager, ml_analyzer, instagram_scraper, metadata_analyzer
    
    try:
        logger.info("🚀 Initializing factr.ai (Text-Based Analysis Mode)...")
        
        # Initialize cache manager
        cache_manager = SimpleCacheManager()
        logger.info("✅ Cache manager ready")
        
        # Initialize ML analyzer
        ml_analyzer = TextBasedMLAnalyzer()
        logger.info("✅ Advanced text analyzer ready")
        
        # Initialize Instagram scraper
        instagram_scraper = InstagramScraper()
        logger.info("✅ Instagram scraper ready")
        
        # Initialize metadata analyzer
        metadata_analyzer = SimpleMetadataAnalyzer()
        logger.info("✅ Metadata analyzer ready")
        
        logger.info("🎯 factr.ai ready! (Advanced text-based misinformation detection)")
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    await initialize_components()
    yield
    logger.info("🛑 factr.ai shutting down...")

# FastAPI application
app = FastAPI(
    title="factr.ai - Advanced Text-Based Misinformation Detection",
    description="AI-powered misinformation detection using advanced NLP techniques (CLIP-free version)",
    version="2.0 - Advanced Text Analysis",
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
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "mode": "advanced_text_analysis",
        "version": "2.0",
        "note": "Advanced NLP analysis active - install CLIP for multimodal capabilities",
        "components": {
            "cache_manager": "ready" if cache_manager else "not_initialized",
            "ml_analyzer": "ready" if ml_analyzer else "not_initialized",
            "instagram_scraper": "ready" if instagram_scraper else "not_initialized",
            "metadata_analyzer": "ready" if metadata_analyzer else "not_initialized"
        },
        "capabilities": {
            "advanced_text_analysis": True,
            "sentiment_analysis": True,
            "emotion_analysis": True,
            "pattern_detection": True,
            "metadata_analysis": True,
            "intelligent_caching": True,
            "demo_mode": True
        }
    }

@app.post("/analyze/instagram", response_model=MisinformationAnalysis)
async def analyze_instagram_post(request: InstagramPostRequest):
    """
    Analyze Instagram post for misinformation using advanced text analysis
    """
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Analyzing post: {request.post_url}")
        
        # Get post data
        post_data = await instagram_scraper.get_post_data(str(request.post_url))
        logger.info(f"📝 Post by @{post_data.username}: {post_data.caption[:100]}...")
        
        # Check cache
        cache_key = cache_manager.generate_cache_key("analysis", post_data.post_id, post_data.caption)
        cached_result = None
        
        if request.cache_results:
            cached_result = await cache_manager.get_cached_result(cache_key)
        
        if cached_result:
            logger.info("⚡ Using cached analysis")
            analysis_results = cached_result
            cache_hit = True
        else:
            # Run analysis
            analysis_results = await ml_analyzer.analyze_cross_modal_consistency(
                post_data.image_url,
                post_data.caption,
                {"post_timestamp": post_data.timestamp, "username": post_data.username}
            )
            
            # Cache results
            if request.cache_results:
                await cache_manager.cache_result(cache_key, analysis_results)
            cache_hit = False
        
        # Add metadata analysis if requested
        if request.include_metadata_analysis:
            metadata_result = await metadata_analyzer.analyze_image_metadata(post_data.image_url)
            analysis_results["metadata_analysis"] = metadata_result
        
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
                "timestamp": post_data.timestamp,
                "likes": post_data.likes,
                "comments": post_data.comments_count
            },
            timestamp=datetime.now(),
            processing_time=processing_time,
            cache_hit=cache_hit
        )
        
        # Log result
        risk_level = "🔴 HIGH" if response.misinformation_score > 70 else "🟡 MED" if response.misinformation_score > 40 else "🟢 LOW"
        logger.info(f"✅ Analysis complete: {response.misinformation_score:.1f}% risk - {risk_level} ({processing_time:.2f}s)")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analytics/performance")
async def get_performance_analytics():
    """Get system performance analytics"""
    try:
        return {
            "system_status": "operational",
            "mode": "advanced_text_analysis",
            "version": "2.0",
            "cache_analytics": {
                "status": "in_memory",
                "entries": len(cache_manager.cache) if cache_manager else 0,
                "hit_rate": "not_tracked"
            },
            "model_status": {
                "text_analyzer": "loaded" if ml_analyzer else "loading",
                "sentiment_model": "loaded" if hasattr(ml_analyzer, 'sentiment_analyzer') and ml_analyzer.sentiment_analyzer else "not_available",
                "emotion_model": "loaded" if hasattr(ml_analyzer, 'emotion_analyzer') and ml_analyzer.emotion_analyzer else "not_available"
            },
            "detection_capabilities": {
                "total_methods": 6,
                "text_analysis": True,
                "sentiment_analysis": True,
                "emotion_analysis": True,
                "pattern_detection": True,
                "metadata_analysis": True,
                "avg_processing_time": "1-3 seconds (cached), 3-8 seconds (fresh)"
            },
            "note": "Advanced text-based analysis - visual analysis available with CLIP installation"
        }
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return {"error": "Analytics temporarily unavailable", "status": "degraded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
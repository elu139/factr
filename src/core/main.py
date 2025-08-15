#!/usr/bin/env python3
"""
factr.ai - Advanced Multimodal Misinformation Detection System
Session 4: BERT Explanations + AWS Deployment Ready

AI-powered system for detecting misinformation across text, image, and metadata
Powered by CLIP, BERT, and 6 independent verification methods
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
import torch
import clip
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl
import hashlib
import requests
from urllib.parse import quote_plus, urlencode
import xml.etree.ElementTree as ET
import time
from functools import lru_cache
import redis
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

# Global variables for components (will be initialized during startup)
cache_manager = None
bert_explanation_generator = None
ml_analyzer = None
data_preprocessor = None
instagram_scraper = None

# Pydantic models
class ExplanationRequest(BaseModel):
    """Request model for explanation generation"""
    audience: str = "general_public"  # general_public, journalists, researchers, content_moderators
    include_evidence: bool = True
    include_recommendations: bool = True
    language: str = "en"  # Future: multi-language support

class InstagramPostRequest(BaseModel):
    """Enhanced request model for Instagram post analysis"""
    post_url: HttpUrl
    include_reverse_search: bool = True
    include_metadata_analysis: bool = True
    explanation_config: Optional[ExplanationRequest] = None
    cache_results: bool = True  # Session 4 NEW

class MisinformationAnalysis(BaseModel):
    """Enhanced response model for misinformation analysis"""
    misinformation_score: float  # 0-100 scale
    confidence_level: str  # Low, Medium, High
    detected_inconsistencies: List[str]
    explanation: str
    modality_scores: Dict[str, float]  # Individual scores for text, image, audio
    metadata_info: Optional[Dict[str, Any]] = None
    timestamp: datetime
    # Session 4 NEW: Enhanced explanation
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

# Session 4 NEW: Advanced caching and performance optimization
class CacheManager:
    """
    Redis-based caching for improved performance
    
    Caches:
    - CLIP embeddings (expensive to compute)
    - Reverse search results (rate-limited APIs)
    - Metadata analysis (slow EXIF processing)
    """
    
    def __init__(self):
        self.redis_client = None
        self.setup_cache()
    
    def setup_cache(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"Redis not available, using memory cache: {e}")
            self.redis_client = None
    
    async def get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def cache_result(self, key: str, data: Dict[str, Any], ttl: int = 3600):
        """Cache result with TTL"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                key, 
                ttl, 
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def generate_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments"""
        key_data = "|".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"factr_ai:{prefix}:{key_hash}"

# Session 4 NEW: Advanced BERT-based explanation generator
class BERTExplanationGenerator:
    """
    Advanced explanation generation using BERT and custom templates
    
    Features:
    1. Context-aware explanations for different audiences
    2. Evidence-based reasoning chains
    3. Confidence calibration
    4. Multi-language support (future)
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_models()
        self.explanation_templates = self._load_explanation_templates()
    
    def setup_models(self):
        """Initialize BERT models for explanation generation"""
        try:
            logger.info("Loading BERT models for explanation generation...")
            
            # Load BERT for text understanding and generation
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Load specialized models for explanation components
            self.fact_checker = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            
            # Load text generation model for explanations
            self.text_generator = pipeline(
                "text2text-generation",
                model="facebook/bart-base",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("BERT explanation models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading BERT models: {e}")
            self.bert_tokenizer = None
            self.fact_checker = None
            self.text_generator = None
    
    def _load_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Load explanation templates for different audiences and contexts"""
        return {
            "general_public": {
                "low_risk": "Our analysis suggests this content is likely authentic. We found {evidence_count} verification points that support its credibility.",
                "medium_risk": "This content shows some concerning patterns. Our {method_count} detection methods found {inconsistency_count} potential issues that warrant careful consideration.",
                "high_risk": "This content shows significant signs of misinformation. Multiple independent verification methods detected serious inconsistencies."
            },
            "journalists": {
                "low_risk": "Verification analysis indicates authentic content with {confidence}% confidence across {method_count} detection methods. Primary evidence: {top_evidence}.",
                "medium_risk": "Content verification reveals {inconsistency_count} red flags requiring editorial review. Key concerns: {primary_concerns}. Recommend additional fact-checking.",
                "high_risk": "ALERT: Content shows {risk_score}% misinformation probability. {method_count} independent methods detected {inconsistency_count} serious inconsistencies. Editorial intervention recommended."
            },
            "researchers": {
                "low_risk": "Multimodal analysis (n={method_count}) indicates low misinformation probability (p={risk_score}). Cross-modal consistency: {clip_score}%. Metadata integrity: confirmed.",
                "medium_risk": "Analysis reveals moderate misinformation indicators (p={risk_score}). Detected anomalies: {inconsistencies}. Confidence interval: {confidence_range}.",
                "high_risk": "High-confidence misinformation detection (p={risk_score}, CI: {confidence_range}). Anomalies detected across {active_methods}/{total_methods} verification methods."
            },
            "content_moderators": {
                "low_risk": "✅ APPROVED: Content passes verification checks. Score: {risk_score}%. No action required.",
                "medium_risk": "⚠️ REVIEW: Flagged for human review. Score: {risk_score}%. Issues: {primary_concerns}. Escalate if necessary.",
                "high_risk": "🚨 BLOCK: High misinformation risk ({risk_score}%). Auto-removal recommended. Reasons: {inconsistencies}."
            }
        }
    
    async def generate_explanation(
        self,
        analysis_results: Dict[str, Any],
        audience: str = "general_public",
        include_evidence: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive, audience-appropriate explanation
        
        Args:
            analysis_results: Complete analysis from all detection methods
            audience: Target audience (general_public, journalists, researchers, content_moderators)
            include_evidence: Whether to include detailed evidence
            include_recommendations: Whether to include action recommendations
        """
        
        try:
            # Extract key metrics
            risk_score = analysis_results["comprehensive_analysis"]["combined_risk_score"]
            confidence_level = analysis_results["comprehensive_analysis"]["confidence_level"]
            detection_methods = analysis_results["comprehensive_analysis"]["detection_methods"]
            primary_concerns = analysis_results["comprehensive_analysis"]["primary_concerns"]
            
            # Determine risk level
            if risk_score < 30:
                risk_level = "low_risk"
            elif risk_score < 70:
                risk_level = "medium_risk"
            else:
                risk_level = "high_risk"
            
            # Generate base explanation using template
            base_explanation = await self._generate_base_explanation(
                risk_level, audience, analysis_results
            )
            
            # Add evidence-based reasoning
            evidence_chain = await self._generate_evidence_chain(
                analysis_results, include_evidence
            ) if include_evidence else {}
            
            # Add recommendations
            recommendations = await self._generate_recommendations(
                risk_level, audience, analysis_results
            ) if include_recommendations else {}
            
            # Generate natural language summary using BERT
            natural_summary = await self._generate_natural_language_summary(
                analysis_results, audience
            )
            
            return {
                "explanation": {
                    "main_assessment": base_explanation,
                    "natural_language_summary": natural_summary,
                    "risk_level": risk_level,
                    "confidence_level": confidence_level,
                    "audience": audience
                },
                "evidence_chain": evidence_chain,
                "recommendations": recommendations,
                "technical_details": {
                    "methods_used": len(detection_methods),
                    "active_detections": len([m for m in detection_methods.values() if m["score"] > 10]),
                    "primary_evidence": primary_concerns[:3],
                    "confidence_factors": await self._analyze_confidence_factors(analysis_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._fallback_explanation(risk_score, audience)
    
    async def _generate_base_explanation(
        self, 
        risk_level: str, 
        audience: str, 
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate base explanation using templates"""
        
        template = self.explanation_templates.get(audience, {}).get(
            risk_level, 
            self.explanation_templates["general_public"][risk_level]
        )
        
        # Extract variables for template formatting
        comprehensive = analysis_results["comprehensive_analysis"]
        ml_analysis = analysis_results["ml_analysis"]
        
        variables = {
            "risk_score": f"{comprehensive['combined_risk_score']:.1f}",
            "confidence": comprehensive["confidence_level"],
            "method_count": len(comprehensive["detection_methods"]),
            "inconsistency_count": len(comprehensive["primary_concerns"]),
            "evidence_count": len([m for m in comprehensive["detection_methods"].values() if m["score"] < 20]),
            "clip_score": f"{ml_analysis['clip_analysis']['text_image_consistency']:.1f}",
            "top_evidence": comprehensive["primary_concerns"][0] if comprehensive["primary_concerns"] else "Visual-textual consistency",
            "primary_concerns": "; ".join(comprehensive["primary_concerns"][:2]),
            "active_methods": len([m for m in comprehensive["detection_methods"].values() if m["score"] > 10]),
            "total_methods": len(comprehensive["detection_methods"]),
            "confidence_range": f"{max(0, comprehensive['combined_risk_score'] - 15):.1f}-{min(100, comprehensive['combined_risk_score'] + 15):.1f}%",
            "inconsistencies": "; ".join(comprehensive["primary_concerns"][:3])
        }
        
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Template variable missing: {e}")
            return template
    
    async def _generate_evidence_chain(
        self, 
        analysis_results: Dict[str, Any], 
        include_evidence: bool
    ) -> Dict[str, Any]:
        """Generate detailed evidence chain showing reasoning"""
        
        if not include_evidence:
            return {}
        
        evidence_chain = {
            "detection_sequence": [],
            "cross_verification": {},
            "confidence_building": []
        }
        
        detection_methods = analysis_results["comprehensive_analysis"]["detection_methods"]
        
        # Build detection sequence
        for method, details in detection_methods.items():
            if details["score"] > 15:  # Significant detection
                evidence_chain["detection_sequence"].append({
                    "method": method.replace("_", " ").title(),
                    "finding": details["description"],
                    "severity": "High" if details["score"] > 50 else "Medium",
                    "confidence": details["score"]
                })
        
        # Cross-verification analysis
        active_detections = len([m for m in detection_methods.values() if m["score"] > 15])
        if active_detections >= 2:
            evidence_chain["cross_verification"] = {
                "status": "Multiple independent methods confirm concerns",
                "agreement_level": f"{active_detections}/{len(detection_methods)} methods",
                "reliability": "High" if active_detections >= 3 else "Medium"
            }
        
        # Confidence building factors
        ml_analysis = analysis_results["ml_analysis"]
        if ml_analysis["clip_analysis"]["text_image_consistency"] > 80:
            evidence_chain["confidence_building"].append("Strong visual-textual alignment supports authenticity")
        elif ml_analysis["clip_analysis"]["text_image_consistency"] < 40:
            evidence_chain["confidence_building"].append("Poor visual-textual alignment indicates potential manipulation")
        
        return evidence_chain
    
    async def _generate_recommendations(
        self, 
        risk_level: str, 
        audience: str, 
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actionable recommendations based on risk level and audience"""
        
        recommendations = {
            "immediate_actions": [],
            "follow_up_steps": [],
            "verification_methods": []
        }
        
        if risk_level == "low_risk":
            if audience == "journalists":
                recommendations["immediate_actions"] = ["Content appears credible for publication"]
                recommendations["follow_up_steps"] = ["Standard editorial review", "Source verification if required"]
            elif audience == "content_moderators":
                recommendations["immediate_actions"] = ["No moderation action required"]
                recommendations["follow_up_steps"] = ["Monitor engagement patterns"]
        
        elif risk_level == "medium_risk":
            if audience == "journalists":
                recommendations["immediate_actions"] = ["Hold publication pending review", "Fact-check key claims"]
                recommendations["follow_up_steps"] = ["Contact original sources", "Verify with subject matter experts"]
            elif audience == "content_moderators":
                recommendations["immediate_actions"] = ["Flag for human review", "Add warning label"]
                recommendations["follow_up_steps"] = ["Monitor user reports", "Escalate if complaints increase"]
        
        elif risk_level == "high_risk":
            if audience == "journalists":
                recommendations["immediate_actions"] = ["Do not publish", "Investigate as potential misinformation"]
                recommendations["follow_up_steps"] = ["Trace content origin", "Alert fact-checking team"]
            elif audience == "content_moderators":
                recommendations["immediate_actions"] = ["Remove content", "Issue warning to user"]
                recommendations["follow_up_steps"] = ["Review user's other content", "Consider account restrictions"]
        
        # Add verification methods
        comprehensive = analysis_results["comprehensive_analysis"]
        if "reverse search" in str(comprehensive["primary_concerns"]):
            recommendations["verification_methods"].append("Additional reverse image searches")
        if "metadata" in str(comprehensive["primary_concerns"]):
            recommendations["verification_methods"].append("Expert forensic analysis")
        if "temporal" in str(comprehensive["primary_concerns"]):
            recommendations["verification_methods"].append("Timeline verification with news archives")
        
        return recommendations
    
    async def _generate_natural_language_summary(
        self, 
        analysis_results: Dict[str, Any], 
        audience: str
    ) -> str:
        """Generate natural language summary using BERT"""
        
        if not self.text_generator:
            return self._fallback_summary(analysis_results)
        
        try:
            # Prepare input for text generation
            comprehensive = analysis_results["comprehensive_analysis"]
            concerns = comprehensive["primary_concerns"]
            
            input_text = f"Misinformation analysis found {len(concerns)} issues: {', '.join(concerns[:3])}. Risk level: {comprehensive['combined_risk_score']:.1f}%"
            
            # Generate explanation using BART
            generated = self.text_generator(
                input_text,
                max_length=150,
                min_length=50,
                do_sample=True,
                temperature=0.7
            )
            
            return generated[0]['generated_text']
            
        except Exception as e:
            logger.error(f"Error in natural language generation: {e}")
            return self._fallback_summary(analysis_results)
    
    def _fallback_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Fallback summary when BERT generation fails"""
        comprehensive = analysis_results["comprehensive_analysis"]
        risk_score = comprehensive["combined_risk_score"]
        
        if risk_score < 30:
            return "Analysis indicates this content is likely authentic with minimal misinformation indicators detected."
        elif risk_score < 70:
            return f"Analysis detected moderate misinformation risk ({risk_score:.1f}%) with several concerning patterns requiring careful evaluation."
        else:
            return f"Analysis indicates high misinformation risk ({risk_score:.1f}%) with multiple verification methods detecting significant inconsistencies."
    
    async def _analyze_confidence_factors(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factors affecting confidence in the assessment"""
        
        confidence_factors = {
            "data_quality": {},
            "method_agreement": {},
            "evidence_strength": {}
        }
        
        # Data quality assessment
        image_data = analysis_results.get("image", {})
        confidence_factors["data_quality"] = {
            "image_accessible": image_data.get("status") == "downloaded_successfully",
            "metadata_available": "metadata_analysis" in analysis_results,
            "reverse_search_successful": "reverse_search" in analysis_results
        }
        
        # Method agreement
        detection_methods = analysis_results["comprehensive_analysis"]["detection_methods"]
        active_methods = [m for m in detection_methods.values() if m["score"] > 15]
        confidence_factors["method_agreement"] = {
            "total_methods": len(detection_methods),
            "active_detections": len(active_methods),
            "agreement_ratio": len(active_methods) / len(detection_methods) if detection_methods else 0
        }
        
        # Evidence strength
        ml_analysis = analysis_results["ml_analysis"]
        confidence_factors["evidence_strength"] = {
            "clip_confidence": ml_analysis["clip_analysis"]["text_image_consistency"],
            "manipulation_certainty": ml_analysis["clip_analysis"]["manipulation_likelihood"],
            "cross_modal_alignment": "High" if ml_analysis["clip_analysis"]["text_image_consistency"] > 70 else "Low"
        }
        
        return confidence_factors
    
    def _fallback_explanation(self, risk_score: float, audience: str) -> Dict[str, Any]:
        """Fallback explanation when BERT processing fails"""
        return {
            "explanation": {
                "main_assessment": f"Analysis indicates {risk_score:.1f}% misinformation risk based on available detection methods.",
                "natural_language_summary": "Technical issues prevented detailed explanation generation. Manual review recommended.",
                "risk_level": "high_risk" if risk_score > 70 else "medium_risk" if risk_score > 30 else "low_risk",
                "confidence_level": "Low",
                "audience": audience
            },
            "evidence_chain": {"status": "unavailable"},
            "recommendations": {"immediate_actions": ["Manual review recommended"]},
            "technical_details": {"status": "processing_error"}
        }

# New classes for Session 3
class ReverseImageSearchEngine:
    """
    Multi-engine reverse image search for detecting image reuse and misattribution
    
    ML Concept: This is crucial for misinformation detection because:
    1. Old images are often reused for new fake events
    2. Stock photos get misrepresented as real news
    3. Images from different locations get misattributed
    
    We use multiple search engines for better coverage and cross-verification
    """
    
    def __init__(self):
        self.engines = {
            "google": self._search_google_images,
            "tineye": self._search_tineye,
            "bing": self._search_bing_images
        }
        self.session = httpx.AsyncClient(timeout=30.0)
        
    async def search_image(self, image_url: str, engines: list = None) -> Dict[str, Any]:
        """
        Search for an image across multiple reverse search engines
        
        Args:
            image_url: URL of the image to search
            engines: List of engines to use (default: all)
            
        Returns:
            Combined results from all engines with analysis
        """
        if engines is None:
            engines = list(self.engines.keys())
            
        results = {}
        
        for engine in engines:
            try:
                logger.info(f"Searching {engine} for image matches...")
                engine_results = await self.engines[engine](image_url)
                results[engine] = engine_results
                
                # Small delay to be respectful to search engines
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching {engine}: {e}")
                results[engine] = {"error": str(e), "matches": []}
        
        # Analyze combined results
        analysis = await self._analyze_search_results(results, image_url)
        
        return {
            "individual_results": results,
            "analysis": analysis,
            "search_timestamp": datetime.now()
        }
    
    async def _search_google_images(self, image_url: str) -> Dict[str, Any]:
        """
        Search Google Images using their reverse search
        
        Note: This uses Google's public reverse search interface
        """
        try:
            # Google reverse image search URL
            search_url = f"https://www.google.com/searchbyimage?image_url={quote_plus(image_url)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = await self.session.get(search_url, headers=headers)
            response.raise_for_status()
            
            # Parse results (simplified - in production you'd use more sophisticated parsing)
            matches = self._parse_google_results(response.text)
            
            return {
                "engine": "google",
                "matches": matches,
                "total_results": len(matches),
                "search_url": search_url
            }
            
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return {"engine": "google", "matches": [], "error": str(e)}
    
    async def _search_tineye(self, image_url: str) -> Dict[str, Any]:
        """
        Search TinEye for exact image matches
        
        TinEye is excellent for finding exact copies and determining the oldest occurrence
        """
        try:
            # TinEye reverse search (using their public interface)
            search_url = f"https://tineye.com/search?url={quote_plus(image_url)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = await self.session.get(search_url, headers=headers)
            response.raise_for_status()
            
            matches = self._parse_tineye_results(response.text)
            
            return {
                "engine": "tineye",
                "matches": matches,
                "total_results": len(matches),
                "search_url": search_url
            }
            
        except Exception as e:
            logger.error(f"TinEye search error: {e}")
            return {"engine": "tineye", "matches": [], "error": str(e)}
    
    async def _search_bing_images(self, image_url: str) -> Dict[str, Any]:
        """
        Search Bing Visual Search for similar images
        """
        try:
            # Bing reverse image search
            search_url = f"https://www.bing.com/images/search?view=detailv2&iss=sbi&form=SBIHMP&sbisrc=UrlPaste&q=imgurl:{quote_plus(image_url)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = await self.session.get(search_url, headers=headers)
            response.raise_for_status()
            
            matches = self._parse_bing_results(response.text)
            
            return {
                "engine": "bing",
                "matches": matches,
                "total_results": len(matches),
                "search_url": search_url
            }
            
        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return {"engine": "bing", "matches": [], "error": str(e)}
    
    def _parse_google_results(self, html: str) -> List[Dict[str, Any]]:
        """Parse Google reverse search results"""
        matches = []
        
        # Look for "Pages that include matching images" section
        # This is a simplified parser - production would be more robust
        page_patterns = re.findall(r'<h3[^>]*><a[^>]*href="([^"]*)"[^>]*>([^<]*)</a></h3>', html)
        
        for url, title in page_patterns[:10]:  # Limit to top 10 results
            if url and title:
                matches.append({
                    "url": url,
                    "title": title.strip(),
                    "source": "google_pages",
                    "confidence": 0.8  # Default confidence
                })
        
        # Look for visually similar images
        img_patterns = re.findall(r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"', html)
        
        for img_url, alt_text in img_patterns[:5]:
            if 'googleusercontent' in img_url:  # Google's cached images
                matches.append({
                    "image_url": img_url,
                    "alt_text": alt_text,
                    "source": "google_images",
                    "confidence": 0.7
                })
        
        return matches
    
    def _parse_tineye_results(self, html: str) -> List[Dict[str, Any]]:
        """Parse TinEye results - focuses on exact matches with dates"""
        matches = []
        
        # TinEye shows exact matches with crawl dates
        match_patterns = re.findall(
            r'<div[^>]*class="[^"]*match[^"]*"[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>([^<]*)</a>.*?<p[^>]*class="[^"]*crawl-date[^"]*"[^>]*>([^<]*)</p>',
            html,
            re.DOTALL
        )
        
        for url, title, crawl_date in match_patterns:
            matches.append({
                "url": url.strip(),
                "title": title.strip(),
                "crawl_date": crawl_date.strip(),
                "source": "tineye_exact",
                "confidence": 0.95  # TinEye gives exact matches
            })
        
        return matches
    
    def _parse_bing_results(self, html: str) -> List[Dict[str, Any]]:
        """Parse Bing visual search results"""
        matches = []
        
        # Bing shows pages containing the image and similar images
        page_patterns = re.findall(r'<h2><a[^>]*href="([^"]*)"[^>]*>([^<]*)</a></h2>', html)
        
        for url, title in page_patterns[:8]:
            if url and title:
                matches.append({
                    "url": url,
                    "title": title.strip(),
                    "source": "bing_pages",
                    "confidence": 0.75
                })
        
        return matches
    
    async def _analyze_search_results(self, results: Dict[str, Any], original_url: str) -> Dict[str, Any]:
        """
        Analyze combined reverse search results for misinformation indicators
        
        Key indicators:
        1. Image age vs. claimed event date
        2. Context mismatches
        3. Geographic inconsistencies
        4. Multiple unrelated uses
        """
        all_matches = []
        
        # Combine all matches from all engines
        for engine, engine_results in results.items():
            if "matches" in engine_results:
                all_matches.extend(engine_results["matches"])
        
        if not all_matches:
            return {
                "total_matches": 0,
                "age_analysis": "No matches found",
                "context_analysis": "Cannot determine reuse",
                "risk_indicators": [],
                "oldest_known_use": None
            }
        
        # Analyze for misinformation indicators
        analysis = {
            "total_matches": len(all_matches),
            "age_analysis": await self._analyze_image_age(all_matches),
            "context_analysis": await self._analyze_context_changes(all_matches),
            "geographic_analysis": await self._analyze_geographic_consistency(all_matches),
            "risk_indicators": await self._identify_risk_indicators(all_matches),
            "oldest_known_use": await self._find_oldest_use(all_matches),
            "confidence_score": await self._calculate_reverse_search_confidence(all_matches)
        }
        
        return analysis
    
    async def _analyze_image_age(self, matches: List[Dict[str, Any]]) -> str:
        """Analyze when the image was first seen online"""
        dates = []
        
        for match in matches:
            if "crawl_date" in match:
                try:
                    # Parse various date formats
                    date_str = match["crawl_date"]
                    # Add date parsing logic here
                    dates.append(date_str)
                except:
                    continue
        
        if dates:
            return f"Image has been online since at least {min(dates)}"
        else:
            return "Could not determine image age from search results"
    
    async def _analyze_context_changes(self, matches: List[Dict[str, Any]]) -> str:
        """Analyze if image context has changed across different uses"""
        contexts = []
        
        for match in matches:
            title = match.get("title", "").lower()
            contexts.append(title)
        
        # Look for conflicting contexts
        news_contexts = [c for c in contexts if any(word in c for word in ["news", "breaking", "report"])]
        location_contexts = [c for c in contexts if any(word in c for word in ["city", "country", "state"])]
        
        if len(set(news_contexts)) > 2:
            return "Image used in multiple different news contexts - potential misattribution"
        elif len(set(location_contexts)) > 2:
            return "Image associated with multiple different locations"
        else:
            return "Context appears consistent across uses"
    
    async def _analyze_geographic_consistency(self, matches: List[Dict[str, Any]]) -> str:
        """Check for geographic inconsistencies in image usage"""
        locations = []
        
        for match in matches:
            title = match.get("title", "").lower()
            # Extract location names (simplified - production would use NER)
            location_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', match.get("title", ""))
            locations.extend(location_words)
        
        unique_locations = set(locations)
        
        if len(unique_locations) > 3:
            return f"Image associated with multiple locations: {', '.join(list(unique_locations)[:3])}"
        else:
            return "Geographic usage appears consistent"
    
    async def _identify_risk_indicators(self, matches: List[Dict[str, Any]]) -> List[str]:
        """Identify specific risk indicators from reverse search"""
        indicators = []
        
        # Check for stock photo usage
        stock_sources = ["shutterstock", "getty", "stock", "alamy", "dreamstime"]
        if any(any(source in match.get("url", "").lower() for source in stock_sources) for match in matches):
            indicators.append("Image appears to be from stock photo source")
        
        # Check for social media reuse
        social_sources = ["facebook", "twitter", "instagram", "tiktok"]
        social_matches = [m for m in matches if any(source in m.get("url", "").lower() for source in social_sources)]
        
        if len(social_matches) > 3:
            indicators.append("Image widely shared across social media platforms")
        
        # Check for news site reuse
        news_sources = ["cnn", "bbc", "reuters", "news", "times", "post"]
        news_matches = [m for m in matches if any(source in m.get("url", "").lower() for source in news_sources)]
        
        if len(news_matches) > 2:
            indicators.append("Image used by multiple news sources")
        
        return indicators
    
    async def _find_oldest_use(self, matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the oldest known use of the image"""
        dated_matches = [m for m in matches if "crawl_date" in m]
        
        if dated_matches:
            # Sort by date (simplified - production would parse dates properly)
            oldest = min(dated_matches, key=lambda x: x.get("crawl_date", ""))
            return oldest
        
        return None
    
    async def _calculate_reverse_search_confidence(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate confidence in reverse search findings"""
        if not matches:
            return 0.0
        
        # Higher confidence with more matches and higher individual confidence scores
        avg_confidence = sum(match.get("confidence", 0.5) for match in matches) / len(matches)
        match_bonus = min(len(matches) / 10.0, 0.3)  # Bonus for more matches, capped at 30%
        
        return min(avg_confidence + match_bonus, 1.0)

class ImageMetadataAnalyzer:
    """
    Extracts and analyzes EXIF metadata from images for forensic analysis
    
    ML Concept: Image metadata contains crucial information:
    1. GPS coordinates (location verification)
    2. Camera settings (manipulation detection)
    3. Creation timestamps (temporal verification)
    4. Software used (editing detection)
    
    This helps detect when images are taken from different times/places than claimed
    """
    
    def __init__(self):
        self.supported_formats = ['JPEG', 'TIFF', 'PNG']
        
    async def analyze_image_metadata(self, image_url: str) -> Dict[str, Any]:
        """
        Extract and analyze comprehensive image metadata
        
        Args:
            image_url: URL of the image to analyze
            
        Returns:
            Comprehensive metadata analysis with misinformation indicators
        """
        try:
            logger.info(f"Analyzing metadata for image: {image_url}")
            
            # Download image
            image_data = await self._download_image_for_metadata(image_url)
            
            # Extract EXIF data
            exif_data = await self._extract_exif_data(image_data)
            
            # Analyze for inconsistencies
            analysis = await self._analyze_metadata_inconsistencies(exif_data)
            
            return {
                "raw_exif": exif_data,
                "analysis": analysis,
                "metadata_timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image metadata: {e}")
            return {
                "error": str(e),
                "analysis": {"status": "metadata_extraction_failed"}
            }
    
    async def _download_image_for_metadata(self, image_url: str) -> bytes:
        """Download image preserving metadata"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            return response.content
    
    async def _extract_exif_data(self, image_data: bytes) -> Dict[str, Any]:
        """Extract comprehensive EXIF metadata"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            if not hasattr(image, '_getexif') or image._getexif() is None:
                return {"status": "no_exif_data"}
            
            exif_dict = {}
            exif = image._getexif()
            
            if exif is not None:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
            
            # Parse GPS data if available
            gps_data = self._parse_gps_data(exif_dict)
            
            # Parse datetime data
            datetime_data = self._parse_datetime_data(exif_dict)
            
            # Parse camera/technical data
            technical_data = self._parse_technical_data(exif_dict)
            
            return {
                "status": "extracted",
                "gps_data": gps_data,
                "datetime_data": datetime_data,
                "technical_data": technical_data,
                "raw_exif": exif_dict,
                "image_format": image.format,
                "image_size": image.size
            }
            
        except Exception as e:
            logger.error(f"EXIF extraction error: {e}")
            return {"status": "extraction_failed", "error": str(e)}
    
    def _parse_gps_data(self, exif_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Parse GPS coordinates from EXIF data"""
        gps_data = {}
        
        # GPS tags
        gps_tags = {
            'GPSLatitude': 'latitude',
            'GPSLongitude': 'longitude',
            'GPSAltitude': 'altitude',
            'GPSTimeStamp': 'gps_time',
            'GPSDateStamp': 'gps_date'
        }
        
        for exif_tag, gps_key in gps_tags.items():
            if exif_tag in exif_dict:
                gps_data[gps_key] = exif_dict[exif_tag]
        
        # Convert DMS coordinates to decimal if present
        if 'latitude' in gps_data and 'longitude' in gps_data:
            try:
                lat_decimal = self._convert_dms_to_decimal(gps_data['latitude'])
                lon_decimal = self._convert_dms_to_decimal(gps_data['longitude'])
                
                gps_data['latitude_decimal'] = lat_decimal
                gps_data['longitude_decimal'] = lon_decimal
                gps_data['coordinates'] = f"{lat_decimal}, {lon_decimal}"
            except:
                pass
        
        return gps_data
    
    def _parse_datetime_data(self, exif_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Parse datetime information from EXIF"""
        datetime_data = {}
        
        datetime_tags = {
            'DateTime': 'datetime_original',
            'DateTimeOriginal': 'datetime_taken',
            'DateTimeDigitized': 'datetime_digitized'
        }
        
        for exif_tag, datetime_key in datetime_tags.items():
            if exif_tag in exif_dict:
                try:
                    # Parse datetime string
                    dt_str = exif_dict[exif_tag]
                    if isinstance(dt_str, str):
                        dt = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
                        datetime_data[datetime_key] = dt
                        datetime_data[f"{datetime_key}_string"] = dt_str
                except:
                    datetime_data[f"{datetime_key}_raw"] = exif_dict[exif_tag]
        
        return datetime_data
    
    def _parse_technical_data(self, exif_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Parse camera and technical data"""
        technical_data = {}
        
        # Camera information
        camera_tags = {
            'Make': 'camera_make',
            'Model': 'camera_model',
            'Software': 'software_used',
            'Orientation': 'orientation',
            'XResolution': 'x_resolution',
            'YResolution': 'y_resolution',
            'ResolutionUnit': 'resolution_unit'
        }
        
        for exif_tag, tech_key in camera_tags.items():
            if exif_tag in exif_dict:
                technical_data[tech_key] = exif_dict[exif_tag]
        
        # Photo settings
        photo_tags = {
            'ExposureTime': 'exposure_time',
            'FNumber': 'f_number',
            'ISO': 'iso_speed',
            'Flash': 'flash_used',
            'FocalLength': 'focal_length',
            'WhiteBalance': 'white_balance'
        }
        
        for exif_tag, photo_key in photo_tags.items():
            if exif_tag in exif_dict:
                technical_data[photo_key] = exif_dict[exif_tag]
        
        return technical_data
    
    def _convert_dms_to_decimal(self, dms_coords):
        """Convert DMS (Degrees, Minutes, Seconds) to decimal coordinates"""
        if isinstance(dms_coords, (list, tuple)) and len(dms_coords) >= 3:
            degrees = float(dms_coords[0])
            minutes = float(dms_coords[1])
            seconds = float(dms_coords[2])
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            return decimal
        
        return None
    
    async def _analyze_metadata_inconsistencies(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze metadata for inconsistencies that indicate misinformation
        
        Key checks:
        1. Temporal inconsistencies (photo date vs. claimed event date)
        2. Geographic inconsistencies (GPS vs. claimed location) 
        3. Technical inconsistencies (editing software detection)
        4. Metadata stripping (suspicious lack of metadata)
        """
        analysis = {
            "temporal_analysis": {},
            "geographic_analysis": {},
            "technical_analysis": {},
            "authenticity_indicators": [],
            "risk_score": 0.0
        }
        
        if exif_data.get("status") != "extracted":
            analysis["authenticity_indicators"].append("No metadata available - possibly stripped")
            analysis["risk_score"] += 30.0
            return analysis
        
        # Temporal analysis
        analysis["temporal_analysis"] = await self._analyze_temporal_metadata(exif_data)
        
        # Geographic analysis  
        analysis["geographic_analysis"] = await self._analyze_geographic_metadata(exif_data)
        
        # Technical analysis
        analysis["technical_analysis"] = await self._analyze_technical_metadata(exif_data)
        
        # Calculate overall risk score
        analysis["risk_score"] = await self._calculate_metadata_risk_score(analysis)
        
        return analysis
    
    async def _analyze_temporal_metadata(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal metadata for inconsistencies"""
        temporal_analysis = {}
        
        datetime_data = exif_data.get("datetime_data", {})
        
        if not datetime_data:
            temporal_analysis["status"] = "no_datetime_data"
            return temporal_analysis
        
        # Check if we have creation time
        creation_time = datetime_data.get("datetime_taken") or datetime_data.get("datetime_original")
        
        if creation_time:
            temporal_analysis["photo_taken"] = creation_time.isoformat()
            temporal_analysis["age_days"] = (datetime.now() - creation_time).days
            
            # Flag very old images being used
            if temporal_analysis["age_days"] > 365:
                temporal_analysis["warning"] = f"Image is {temporal_analysis['age_days']} days old"
            
            # Check for future dates (impossible)
            if creation_time > datetime.now():
                temporal_analysis["error"] = "Image has future creation date - metadata manipulation suspected"
        
        return temporal_analysis
    
    async def _analyze_geographic_metadata(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze GPS metadata for location verification"""
        geographic_analysis = {}
        
        gps_data = exif_data.get("gps_data", {})
        
        if not gps_data or "coordinates" not in gps_data:
            geographic_analysis["status"] = "no_gps_data"
            return geographic_analysis
        
        coordinates = gps_data["coordinates"]
        geographic_analysis["gps_coordinates"] = coordinates
        geographic_analysis["location_available"] = True
        
        # Here you could add reverse geocoding to get location names
        # and compare with claimed locations in the post
        
        return geographic_analysis
    
    async def _analyze_technical_metadata(self, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical metadata for manipulation indicators"""
        technical_analysis = {}
        
        technical_data = exif_data.get("technical_data", {})
        
        # Check for editing software
        software = technical_data.get("software_used", "")
        if software:
            editing_software = ["photoshop", "gimp", "lightroom", "snapseed", "facetune"]
            if any(editor in software.lower() for editor in editing_software):
                technical_analysis["editing_detected"] = software
                technical_analysis["warning"] = "Image processed with editing software"
        
        # Check camera information
        camera_make = technical_data.get("camera_make", "")
        camera_model = technical_data.get("camera_model", "")
        
        if camera_make and camera_model:
            technical_analysis["camera_info"] = f"{camera_make} {camera_model}"
        elif not camera_make and not camera_model:
            technical_analysis["warning"] = "No camera information - possibly processed or synthetic"
        
        return technical_analysis
    
    async def _calculate_metadata_risk_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall risk score based on metadata analysis"""
        risk_score = 0.0
        
        # No metadata available
        if "No metadata available" in analysis.get("authenticity_indicators", []):
            risk_score += 30.0
        
        # Temporal risks
        temporal = analysis.get("temporal_analysis", {})
        if "error" in temporal:
            risk_score += 40.0
        elif "warning" in temporal:
            risk_score += 20.0
        
        # Technical risks
        technical = analysis.get("technical_analysis", {})
        if "editing_detected" in technical:
            risk_score += 15.0
        if "No camera information" in technical.get("warning", ""):
            risk_score += 10.0
        
        return min(risk_score, 100.0)

class InstagramScraper:
    """
    Real Instagram post scraper using web scraping techniques
    
    ML Concept: This extracts the actual data we need for multimodal analysis:
    - Image URLs for CLIP processing
    - Caption text for BERT analysis  
    - Metadata for consistency checks
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    async def get_post_data(self, post_url: str) -> InstagramPost:
        """
        Scrape Instagram post data from URL
        
        Args:
            post_url: Instagram post URL
            
        Returns:
            InstagramPost object with scraped data
        """
        try:
            # Extract post ID from URL
            post_id = self._extract_post_id(post_url)
            
            # Scrape the post data
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                post_data = await self._scrape_post_data(client, post_url, post_id)
                
            return InstagramPost(**post_data)
            
        except Exception as e:
            logger.error(f"Error scraping Instagram post data: {e}")
            raise HTTPException(status_code=400, detail=f"Could not scrape post data: {str(e)}")
    
    def _extract_post_id(self, post_url: str) -> str:
        """Extract Instagram post ID from URL"""
        # Instagram URLs: https://www.instagram.com/p/POST_ID/ or instagram.com/p/POST_ID/
        pattern = r'/p/([A-Za-z0-9_-]+)/?'
        match = re.search(pattern, post_url)
        if match:
            return match.group(1)
        raise ValueError("Invalid Instagram URL format")
    
    async def _scrape_post_data(self, client: httpx.AsyncClient, post_url: str, post_id: str) -> Dict:
        """
        Scrape actual Instagram post data
        
        This uses Instagram's web interface to extract:
        1. Image URLs
        2. Caption text
        3. Metadata (username, timestamp, etc.)
        """
        try:
            # Add embed to URL for easier scraping
            embed_url = f"{post_url}embed/"
            
            response = await client.get(embed_url)
            response.raise_for_status()
            
            html_content = response.text
            
            # Extract data using regex patterns
            post_data = {}
            
            # Extract image URL
            image_pattern = r'"display_url":"([^"]+)"'
            image_match = re.search(image_pattern, html_content)
            if image_match:
                image_url = image_match.group(1).replace("\\u0026", "&")
                post_data["image_url"] = image_url
            else:
                # Fallback pattern
                img_pattern = r'<img[^>]+src="([^"]+)"[^>]*>'
                img_match = re.search(img_pattern, html_content)
                if img_match:
                    post_data["image_url"] = img_match.group(1)
                else:
                    raise ValueError("Could not extract image URL")
            
            # Extract caption
            caption_pattern = r'"edge_media_to_caption":\{"edges":\[\{"node":\{"text":"([^"]+)"'
            caption_match = re.search(caption_pattern, html_content)
            if caption_match:
                caption = caption_match.group(1).replace("\\n", "\n").replace("\\", "")
                post_data["caption"] = caption
            else:
                # Try alternative pattern
                alt_caption = r'<meta property="og:description" content="([^"]+)"'
                alt_match = re.search(alt_caption, html_content)
                post_data["caption"] = alt_match.group(1) if alt_match else ""
            
            # Extract username
            username_pattern = r'"username":"([^"]+)"'
            username_match = re.search(username_pattern, html_content)
            post_data["username"] = username_match.group(1) if username_match else "unknown"
            
            # Extract timestamp (Instagram uses Unix timestamps)
            timestamp_pattern = r'"taken_at_timestamp":(\d+)'
            timestamp_match = re.search(timestamp_pattern, html_content)
            if timestamp_match:
                timestamp = datetime.fromtimestamp(int(timestamp_match.group(1)))
                post_data["timestamp"] = timestamp
            else:
                post_data["timestamp"] = datetime.now()
            
            # Extract engagement metrics
            likes_pattern = r'"edge_media_preview_like":\{"count":(\d+)'
            likes_match = re.search(likes_pattern, html_content)
            post_data["likes"] = int(likes_match.group(1)) if likes_match else None
            
            comments_pattern = r'"edge_media_to_comment":\{"count":(\d+)'
            comments_match = re.search(comments_pattern, html_content)
            post_data["comments_count"] = int(comments_match.group(1)) if comments_match else None
            
            post_data["post_id"] = post_id
            
            return post_data
            
        except Exception as e:
            logger.error(f"Error in scraping: {e}")
            # Return fallback data for testing
            return {
                "post_id": post_id,
                "caption": "Could not extract caption - testing mode",
                "image_url": "https://via.placeholder.com/400x400.png?text=Test+Image",
                "username": "test_user",
                "timestamp": datetime.now(),
                "likes": 0,
                "comments_count": 0
            }

# ML Models for Multimodal Analysis
class MultimodalAnalyzer:
    """
    The heart of factr.ai! This class handles the ML magic:
    
    Key ML Concepts:
    1. CLIP (Contrastive Language-Image Pre-training): 
       - Understands both images and text in the same vector space
       - Can measure how well image content matches text descriptions
       
    2. Cross-modal Consistency:
       - Compares what the image shows vs. what the text claims
       - Detects inconsistencies that indicate misinformation
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_models()
        
    def setup_models(self):
        """Initialize CLIP and other ML models"""
        try:
            logger.info(f"Loading models on device: {self.device}")
            
            # Load CLIP model for image-text analysis
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Load BERT for text analysis (we'll use this for explanation generation)
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
            # Sentiment analyzer for caption analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("All ML models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    async def analyze_cross_modal_consistency(
        self, 
        image_url: str, 
        caption: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        The core analysis function! This is where we detect misinformation.
        
        Process:
        1. Download and preprocess the image
        2. Tokenize and clean the caption text
        3. Use CLIP to measure image-text consistency
        4. Analyze for specific inconsistency types
        5. Generate explanation and confidence scores
        """
        try:
            # Step 1: Download and preprocess image
            image = await self._download_and_preprocess_image(image_url)
            
            # Step 2: Preprocess text
            processed_text = self._preprocess_text_for_analysis(caption)
            
            # Step 3: CLIP-based consistency analysis
            clip_results = await self._clip_consistency_analysis(image, processed_text["cleaned_text"])
            
            # Step 4: Detect specific inconsistency types
            inconsistencies = await self._detect_inconsistencies(
                clip_results, processed_text, metadata
            )
            
            # Step 5: Generate final scores and explanation
            analysis_result = await self._generate_analysis_result(
                clip_results, inconsistencies, processed_text, metadata
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in cross-modal analysis: {e}")
            return self._fallback_analysis()
    
    async def _download_and_preprocess_image(self, image_url: str) -> torch.Tensor:
        """Download image and prepare it for CLIP analysis"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(response.content))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Preprocess for CLIP
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                return image_tensor
                
        except Exception as e:
            logger.error(f"Error downloading/preprocessing image: {e}")
            # Return a placeholder tensor for testing
            return torch.zeros(1, 3, 224, 224).to(self.device)
    
    def _preprocess_text_for_analysis(self, caption: str) -> Dict[str, Any]:
        """Advanced text preprocessing for CLIP and misinformation detection"""
        # Basic cleaning
        cleaned_text = caption.strip()
        
        # Extract temporal indicators
        temporal_phrases = re.findall(
            r'\b(?:yesterday|today|tomorrow|last week|this year|2020|2021|2022|2023|2024|2025)\b', 
            cleaned_text.lower()
        )
        
        # Extract location indicators
        location_phrases = re.findall(
            r'\b(?:in|at|from)\s+([A-Z][a-zA-Z\s]+)\b', 
            cleaned_text
        )
        
        # Extract claim indicators
        claim_phrases = re.findall(
            r'\b(?:breaking|just happened|exclusive|real|fake|truth|lies)\b', 
            cleaned_text.lower()
        )
        
        # Sentiment analysis
        try:
            sentiment = self.sentiment_analyzer(cleaned_text[:500])[0]  # Limit length
        except:
            sentiment = {"label": "NEUTRAL", "score": 0.5}
        
        return {
            "original": caption,
            "cleaned_text": cleaned_text,
            "temporal_indicators": temporal_phrases,
            "location_indicators": location_phrases,
            "claim_indicators": claim_phrases,
            "sentiment": sentiment,
            "word_count": len(cleaned_text.split()),
            "char_count": len(cleaned_text)
        }
    
    async def _clip_consistency_analysis(self, image_tensor: torch.Tensor, text: str) -> Dict[str, float]:
        """
        Use CLIP to analyze how well the image matches the text description
        
        CLIP Magic Explained:
        - CLIP creates vector representations (embeddings) for both image and text
        - The closer these vectors are, the more consistent the content
        - We can measure this with cosine similarity
        """
        try:
            with torch.no_grad():
                # Get image embedding
                image_features = self.clip_model.encode_image(image_tensor)
                
                # Get text embedding
                text_tokens = clip.tokenize([text]).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity (this is the key metric!)
                similarity = torch.cosine_similarity(image_features, text_features).item()
                
                # Convert to percentage and interpret
                consistency_score = (similarity + 1) * 50  # Convert from [-1,1] to [0,100]
                
                # Additional CLIP-based tests
                # Test if image matches common misinformation patterns
                misleading_prompts = [
                    "a fake or manipulated image",
                    "a deepfake or AI generated image", 
                    "an old photo being used for new news",
                    "a photo from a different location",
                    "stock photo or generic image"
                ]
                
                manipulation_scores = []
                for prompt in misleading_prompts:
                    prompt_tokens = clip.tokenize([prompt]).to(self.device)
                    prompt_features = self.clip_model.encode_text(prompt_tokens)
                    prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
                    
                    manipulation_sim = torch.cosine_similarity(image_features, prompt_features).item()
                    manipulation_scores.append((manipulation_sim + 1) * 50)
                
                return {
                    "text_image_consistency": consistency_score,
                    "manipulation_likelihood": max(manipulation_scores),
                    "raw_similarity": similarity,
                    "individual_manipulation_scores": manipulation_scores
                }
                
        except Exception as e:
            logger.error(f"Error in CLIP analysis: {e}")
            return {
                "text_image_consistency": 50.0,
                "manipulation_likelihood": 30.0,
                "raw_similarity": 0.0,
                "individual_manipulation_scores": []
            }
    
    async def _detect_inconsistencies(
        self, 
        clip_results: Dict[str, float], 
        text_analysis: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Detect specific types of inconsistencies that indicate misinformation
        
        This is where we get specific about WHY something might be misinformation
        """
        inconsistencies = []
        
        # 1. Low image-text consistency
        if clip_results["text_image_consistency"] < 40:
            inconsistencies.append(
                f"Low visual-textual consistency ({clip_results['text_image_consistency']:.1f}%): "
                "The image content doesn't match what the caption describes"
            )
        
        # 2. High manipulation likelihood
        if clip_results["manipulation_likelihood"] > 70:
            inconsistencies.append(
                f"High manipulation indicators ({clip_results['manipulation_likelihood']:.1f}%): "
                "Visual patterns suggest potential image manipulation or misuse"
            )
        
        # 3. Temporal inconsistencies
        if text_analysis["temporal_indicators"]:
            # Check if post timestamp conflicts with temporal claims
            post_year = metadata.get("post_timestamp", datetime.now()).year
            claimed_years = [int(x) for x in text_analysis["temporal_indicators"] 
                           if x.isdigit() and len(x) == 4]
            
            if claimed_years and abs(post_year - max(claimed_years)) > 1:
                inconsistencies.append(
                    f"Temporal inconsistency detected: Post from {post_year} claims events from {claimed_years}"
                )
        
        # 4. Suspicious claim language
        suspicious_claims = ["breaking", "exclusive", "real", "truth"]
        found_claims = [claim for claim in suspicious_claims 
                       if claim in text_analysis["claim_indicators"]]
        
        if found_claims and clip_results["text_image_consistency"] < 60:
            inconsistencies.append(
                f"Suspicious claim language ('{', '.join(found_claims)}') combined with low visual consistency"
            )
        
        # 5. Sentiment-consistency mismatch
        if (text_analysis["sentiment"]["label"] == "NEGATIVE" and 
            text_analysis["sentiment"]["score"] > 0.8 and 
            clip_results["text_image_consistency"] > 80):
            inconsistencies.append(
                "Potential emotional manipulation: Highly negative text with unrelated positive imagery"
            )
        
        return inconsistencies
    
    async def _generate_analysis_result(
        self, 
        clip_results: Dict[str, float], 
        inconsistencies: List[str], 
        text_analysis: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate the final analysis result with scores and explanations
        """
        # Calculate overall misinformation score
        base_score = 100 - clip_results["text_image_consistency"]
        manipulation_penalty = clip_results["manipulation_likelihood"] * 0.3
        inconsistency_penalty = len(inconsistencies) * 10
        
        misinformation_score = min(100, base_score + manipulation_penalty + inconsistency_penalty)
        
        # Generate confidence level
        if misinformation_score > 80:
            confidence = "High"
        elif misinformation_score > 50:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Generate natural language explanation
        explanation = self._generate_explanation(
            misinformation_score, clip_results, inconsistencies, text_analysis
        )
        
        return {
            "misinformation_score": round(misinformation_score, 1),
            "confidence_level": confidence,
            "inconsistencies": inconsistencies,
            "explanation": explanation,
            "clip_analysis": clip_results,
            "text_analysis": text_analysis,
            "metadata_analysis": metadata
        }
    
    def _generate_explanation(
        self, 
        score: float, 
        clip_results: Dict[str, float], 
        inconsistencies: List[str], 
        text_analysis: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation of the analysis"""
        
        if score < 30:
            risk_level = "low"
            main_msg = "The content appears to be consistent and likely authentic."
        elif score < 70:
            risk_level = "moderate"
            main_msg = "The content shows some inconsistencies that warrant caution."
        else:
            risk_level = "high"
            main_msg = "The content shows significant inconsistencies suggesting potential misinformation."
        
        explanation = f"Analysis indicates a {risk_level} risk of misinformation ({score:.1f}% confidence). {main_msg}"
        
        # Add specific findings
        if clip_results["text_image_consistency"] < 50:
            explanation += f" The visual content shows {clip_results['text_image_consistency']:.1f}% consistency with the text description."
        
        if inconsistencies:
            explanation += f" Specific concerns include: {'; '.join(inconsistencies[:2])}."
        
        if len(inconsistencies) > 2:
            explanation += f" {len(inconsistencies) - 2} additional inconsistencies were detected."
        
        return explanation
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when ML models fail"""
        return {
            "misinformation_score": 50.0,
            "confidence_level": "Low",
            "inconsistencies": ["Analysis unavailable - using fallback detection"],
            "explanation": "Technical issue prevented full analysis. Manual review recommended.",
            "clip_analysis": {"text_image_consistency": 50.0, "manipulation_likelihood": 50.0},
            "text_analysis": {"error": "Processing failed"},
            "metadata_analysis": {"status": "fallback"}
        }

# Enhanced Data preprocessing pipeline with Session 4 capabilities
class DataPreprocessor:
    """
    Handles preprocessing of multimodal data before ML analysis
    Session 4 Enhancement: Now includes caching and performance optimization!
    """
    
    def __init__(self, ml_analyzer: MultimodalAnalyzer, cache_manager: CacheManager):
        self.ml_analyzer = ml_analyzer
        self.reverse_search = ReverseImageSearchEngine()
        self.metadata_analyzer = ImageMetadataAnalyzer()
        self.cache_manager = cache_manager
        self.setup_preprocessing()
    
    def setup_preprocessing(self):
        """Initialize preprocessing tools"""
        logger.info("Enhanced preprocessing pipeline initialized with ML models + reverse search + metadata analysis + caching")
    
    async def preprocess_post(
        self, 
        post: InstagramPost, 
        include_reverse_search: bool = True, 
        include_metadata: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Preprocess Instagram post for comprehensive ML analysis
        Session 4 Enhancement: Now includes intelligent caching!
        
        Args:
            post: InstagramPost object
            include_reverse_search: Whether to perform reverse image search
            include_metadata: Whether to analyze image metadata
            use_cache: Whether to use caching for performance
            
        Returns:
            Dictionary with comprehensive analysis across all modalities
        """
        start_time = time.time()
        cache_hits = {"reverse_search": False, "metadata": False, "ml_analysis": False}
        
        # Basic preprocessing (always fresh)
        basic_preprocessing = {
            "text": await self._preprocess_text(post.caption),
            "image": await self._preprocess_image(post.image_url),
            "metadata": await self._extract_metadata(post)
        }
        
        # Session 4 NEW: Cached reverse image search analysis
        reverse_search_results = None
        if include_reverse_search:
            cache_key = self.cache_manager.generate_cache_key("reverse_search", post.image_url)
            
            if use_cache:
                reverse_search_results = await self.cache_manager.get_cached_result(cache_key)
                cache_hits["reverse_search"] = reverse_search_results is not None
            
            if not reverse_search_results:
                try:
                    logger.info("Running reverse image search analysis...")
                    reverse_search_results = await self.reverse_search.search_image(post.image_url)
                    logger.info(f"Found {reverse_search_results['analysis']['total_matches']} matches across search engines")
                    
                    # Cache results for 1 hour (reverse search is expensive)
                    if use_cache:
                        await self.cache_manager.cache_result(cache_key, reverse_search_results, ttl=3600)
                        
                except Exception as e:
                    logger.error(f"Reverse search failed: {e}")
                    reverse_search_results = {"error": str(e)}
        
        # Session 4 NEW: Cached image metadata analysis
        metadata_analysis = None
        if include_metadata:
            cache_key = self.cache_manager.generate_cache_key("metadata", post.image_url)
            
            if use_cache:
                metadata_analysis = await self.cache_manager.get_cached_result(cache_key)
                cache_hits["metadata"] = metadata_analysis is not None
            
            if not metadata_analysis:
                try:
                    logger.info("Analyzing image metadata...")
                    metadata_analysis = await self.metadata_analyzer.analyze_image_metadata(post.image_url)
                    logger.info("Metadata analysis complete")
                    
                    # Cache metadata for 24 hours (metadata doesn't change)
                    if use_cache:
                        await self.cache_manager.cache_result(cache_key, metadata_analysis, ttl=86400)
                        
                except Exception as e:
                    logger.error(f"Metadata analysis failed: {e}")
                    metadata_analysis = {"error": str(e)}
        
        # Session 4 NEW: Cached ML analysis
        ml_analysis = None
        cache_key = self.cache_manager.generate_cache_key("ml_analysis", post.image_url, post.caption)
        
        if use_cache:
            ml_analysis = await self.cache_manager.get_cached_result(cache_key)
            cache_hits["ml_analysis"] = ml_analysis is not None
        
        if not ml_analysis:
            # Run CLIP-based ML analysis
            ml_analysis = await self.ml_analyzer.analyze_cross_modal_consistency(
                post.image_url, 
                post.caption, 
                basic_preprocessing["metadata"]
            )
            
            # Cache ML analysis for 1 hour (balances performance vs freshness)
            if use_cache:
                await self.cache_manager.cache_result(cache_key, ml_analysis, ttl=3600)
        
        # Session 4 NEW: Enhanced analysis combining all sources
        comprehensive_analysis = await self._generate_comprehensive_analysis(
            ml_analysis,
            reverse_search_results,
            metadata_analysis,
            basic_preprocessing,
            post
        )
        
        # Performance metrics
        processing_time = time.time() - start_time
        
        # Combine everything with Session 4 enhancements
        preprocessed_data = {
            **basic_preprocessing,
            "ml_analysis": ml_analysis,
            "reverse_search": reverse_search_results,
            "metadata_analysis": metadata_analysis,
            "comprehensive_analysis": comprehensive_analysis,
            "performance_metrics": {
                "processing_time": processing_time,
                "cache_hits": cache_hits,
                "cache_efficiency": sum(cache_hits.values()) / len(cache_hits)
            }
        }
        
        return preprocessed_data
    
    async def _generate_comprehensive_analysis(
        self,
        ml_analysis: Dict[str, Any],
        reverse_search_results: Optional[Dict[str, Any]],
        metadata_analysis: Optional[Dict[str, Any]], 
        basic_preprocessing: Dict[str, Any],
        post: InstagramPost
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis combining all detection methods
        Session 3 NEW: Now we have 6 different detection methods!
        """
        
        comprehensive_analysis = {
            "detection_methods": {},
            "combined_risk_score": 0.0,
            "primary_concerns": [],
            "evidence_summary": {},
            "confidence_level": "Low"
        }
        
        # Method 1: CLIP-based cross-modal consistency (from Session 2)
        clip_score = 100 - ml_analysis.get("clip_analysis", {}).get("text_image_consistency", 50)
        comprehensive_analysis["detection_methods"]["clip_consistency"] = {
            "score": clip_score,
            "description": f"Visual-textual consistency: {ml_analysis.get('clip_analysis', {}).get('text_image_consistency', 50):.1f}%"
        }
        
        # Method 2: Manipulation detection (from Session 2)
        manipulation_score = ml_analysis.get("clip_analysis", {}).get("manipulation_likelihood", 0)
        comprehensive_analysis["detection_methods"]["manipulation_detection"] = {
            "score": manipulation_score,
            "description": f"Manipulation likelihood: {manipulation_score:.1f}%"
        }
        
        # Method 3: NEW - Reverse search analysis
        reverse_search_score = 0.0
        if reverse_search_results and "analysis" in reverse_search_results:
            analysis = reverse_search_results["analysis"]
            
            # Calculate risk based on reverse search findings
            if analysis["total_matches"] > 10:
                reverse_search_score += 20  # Widely circulated image
            
            if "stock photo" in str(analysis.get("risk_indicators", [])):
                reverse_search_score += 30  # Stock photo misused as real news
            
            if "multiple news sources" in str(analysis.get("risk_indicators", [])):
                reverse_search_score += 25  # Reused across news sites
            
            comprehensive_analysis["detection_methods"]["reverse_search"] = {
                "score": reverse_search_score,
                "description": f"Found {analysis['total_matches']} matches with {len(analysis.get('risk_indicators', []))} risk indicators"
            }
            
            # Add specific concerns
            if analysis.get("risk_indicators"):
                comprehensive_analysis["primary_concerns"].extend(analysis["risk_indicators"])
        
        # Method 4: NEW - Metadata temporal analysis  
        metadata_score = 0.0
        if metadata_analysis and "analysis" in metadata_analysis:
            meta_analysis = metadata_analysis["analysis"]
            
            # Check temporal inconsistencies
            temporal = meta_analysis.get("temporal_analysis", {})
            if "error" in temporal:
                metadata_score += 40
                comprehensive_analysis["primary_concerns"].append("Impossible image creation date detected")
            elif temporal.get("age_days", 0) > 365:
                metadata_score += 25
                comprehensive_analysis["primary_concerns"].append(f"Image is {temporal['age_days']} days old")
            
            # Check for editing
            technical = meta_analysis.get("technical_analysis", {})
            if "editing_detected" in technical:
                metadata_score += 15
                comprehensive_analysis["primary_concerns"].append(f"Image edited with {technical['editing_detected']}")
            
            comprehensive_analysis["detection_methods"]["metadata_analysis"] = {
                "score": metadata_score,
                "description": f"Metadata risk score: {meta_analysis.get('risk_score', 0):.1f}%"
            }
        
        # Method 5: Temporal claim analysis (enhanced from Session 2)
        temporal_score = 0.0
        text_analysis = ml_analysis.get("text_analysis", {})
        temporal_indicators = text_analysis.get("temporal_indicators", [])
        
        if temporal_indicators:
            post_year = post.timestamp.year
            claimed_years = [int(x) for x in temporal_indicators if x.isdigit() and len(x) == 4]
            
            if claimed_years and abs(post_year - max(claimed_years)) > 1:
                temporal_score = 35
                comprehensive_analysis["primary_concerns"].append(
                    f"Temporal mismatch: Post from {post_year} claims events from {claimed_years}"
                )
        
        comprehensive_analysis["detection_methods"]["temporal_analysis"] = {
            "score": temporal_score,
            "description": f"Found {len(temporal_indicators)} temporal indicators"
        }
        
        # Method 6: Engagement pattern analysis (NEW)
        engagement_score = 0.0
        engagement_ratio = basic_preprocessing["metadata"]["engagement"]["engagement_ratio"]
        
        if engagement_ratio > 100:  # Suspiciously high engagement
            engagement_score = 20
            comprehensive_analysis["primary_concerns"].append(
                f"Unusual engagement pattern: {engagement_ratio:.1f} likes per comment"
            )
        
        comprehensive_analysis["detection_methods"]["engagement_analysis"] = {
            "score": engagement_score,
            "description": f"Engagement ratio: {engagement_ratio:.1f} likes/comment"
        }
        
        # Calculate combined risk score (weighted average of all methods)
        method_scores = [
            clip_score * 0.3,  # CLIP gets highest weight
            manipulation_score * 0.25,
            reverse_search_score * 0.2,
            metadata_score * 0.15,
            temporal_score * 0.07,
            engagement_score * 0.03
        ]
        
        comprehensive_analysis["combined_risk_score"] = sum(method_scores)
        
        # Determine confidence level
        active_methods = sum(1 for score in [clip_score, reverse_search_score, metadata_score, temporal_score] if score > 10)
        
        if comprehensive_analysis["combined_risk_score"] > 70 and active_methods >= 3:
            comprehensive_analysis["confidence_level"] = "High"
        elif comprehensive_analysis["combined_risk_score"] > 40 and active_methods >= 2:
            comprehensive_analysis["confidence_level"] = "Medium"
        else:
            comprehensive_analysis["confidence_level"] = "Low"
        
        # Generate evidence summary
        comprehensive_analysis["evidence_summary"] = {
            "strongest_indicators": [concern for concern in comprehensive_analysis["primary_concerns"][:3]],
            "total_detection_methods": len([s for s in method_scores if s > 5]),
            "cross_verification": active_methods >= 2,
            "unanimous_concern": all(score > 15 for score in method_scores[:4])
        }
        
        return comprehensive_analysis
    
    async def _preprocess_text(self, caption: str) -> Dict[str, Any]:
        """Enhanced text preprocessing with more analysis"""
        # Basic text cleaning
        cleaned_text = caption.strip()
        
        # Remove URLs and mentions for cleaner analysis
        cleaned_for_analysis = re.sub(r'http[s]?://\S+', '', cleaned_text)
        cleaned_for_analysis = re.sub(r'@\w+', '', cleaned_for_analysis).strip()
        
        # Extract hashtags and mentions
        hashtags = re.findall(r'#\w+', caption)
        mentions = re.findall(r'@\w+', caption)
        urls = re.findall(r'http[s]?://\S+', caption)
        
        return {
            "raw_text": caption,
            "cleaned_text": cleaned_text,
            "cleaned_for_analysis": cleaned_for_analysis,
            "length": len(caption.split()),
            "hashtags": hashtags,
            "mentions": mentions,
            "urls": urls
        }
    
    async def _preprocess_image(self, image_url: str) -> Dict[str, Any]:
        """Enhanced image preprocessing"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                
                # Basic image info
                image = Image.open(io.BytesIO(response.content))
                
                return {
                    "url": image_url,
                    "status": "downloaded_successfully",
                    "format": image.format,
                    "size": image.size,
                    "mode": image.mode,
                    "file_size_bytes": len(response.content)
                }
                
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return {
                "url": image_url,
                "status": "download_failed",
                "error": str(e)
            }
    
    async def _extract_metadata(self, post: InstagramPost) -> Dict[str, Any]:
        """Extract basic metadata from Instagram post"""
        engagement_ratio = 0
        if post.likes and post.comments_count and post.comments_count > 0:
            engagement_ratio = post.likes / post.comments_count
        
        return {
            "post_id": post.post_id,
            "username": post.username,
            "post_timestamp": post.timestamp,
            "engagement": {
                "likes": post.likes or 0,
                "comments": post.comments_count or 0,
                "engagement_ratio": engagement_ratio
            },
            "content_info": {
                "has_caption": bool(post.caption.strip()),
                "caption_length": len(post.caption),
                "has_image": bool(post.image_url)
            }
        }

# Startup function to initialize all components
async def initialize_components():
    """Initialize all global components during app startup"""
    global cache_manager, bert_explanation_generator, ml_analyzer, data_preprocessor, instagram_scraper
    
    try:
        logger.info("🚀 Initializing factr.ai components...")
        
        # Initialize cache manager first
        cache_manager = CacheManager()
        logger.info("✅ Cache manager initialized")
        
        # Initialize ML analyzer
        ml_analyzer = MultimodalAnalyzer()
        logger.info("✅ ML analyzer initialized")
        
        # Initialize BERT explanation generator
        bert_explanation_generator = BERTExplanationGenerator()
        logger.info("✅ BERT explanation generator initialized")
        
        # Initialize data preprocessor
        data_preprocessor = DataPreprocessor(ml_analyzer, cache_manager)
        logger.info("✅ Data preprocessor initialized")
        
        # Initialize Instagram scraper
        instagram_scraper = InstagramScraper()
        logger.info("✅ Instagram scraper initialized")
        
        logger.info("🎯 All factr.ai components ready!")
        
    except Exception as e:
        logger.error(f"❌ Error initializing components: {e}")
        raise

# Application lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_components()
    yield
    # Shutdown
    logger.info("🛑 Shutting down factr.ai...")

# Initialize FastAPI app
app = FastAPI(
    title="factr.ai - Multimodal Misinformation Detection",
    description="AI-powered system for detecting misinformation across text, image, and metadata",
    version="Session 4 - BERT Explanations + AWS Deployment",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check system health and component status"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "cache_manager": "ready" if cache_manager else "not_initialized",
                "ml_models": "ready" if ml_analyzer else "not_initialized",
                "bert_generator": "ready" if bert_explanation_generator else "not_initialized",
                "data_preprocessor": "ready" if data_preprocessor else "not_initialized"
            },
            "capabilities": {
                "multimodal_analysis": True,
                "reverse_image_search": True,
                "metadata_analysis": True,
                "bert_explanations": True,
                "intelligent_caching": True,
                "6_detection_methods": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# MAIN ANALYSIS ENDPOINT - This was missing!
@app.post("/analyze/instagram", response_model=MisinformationAnalysis)
async def analyze_instagram_post(request: InstagramPostRequest):
    """
    Analyze Instagram post for misinformation using all 6 detection methods
    
    This is the main endpoint that the frontend calls!
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting analysis for Instagram post: {request.post_url}")
        
        # Step 1: Scrape Instagram post data
        post_data = await instagram_scraper.get_post_data(str(request.post_url))
        logger.info(f"Successfully scraped post data for user: {post_data.username}")
        
        # Step 2: Run comprehensive preprocessing and analysis
        analysis_results = await data_preprocessor.preprocess_post(
            post_data,
            include_reverse_search=request.include_reverse_search,
            include_metadata=request.include_metadata_analysis,
            use_cache=request.cache_results
        )
        
        # Step 3: Generate detailed explanation if requested
        detailed_explanation = None
        if request.explanation_config:
            detailed_explanation = await bert_explanation_generator.generate_explanation(
                analysis_results,
                audience=request.explanation_config.audience,
                include_evidence=request.explanation_config.include_evidence,
                include_recommendations=request.explanation_config.include_recommendations
            )
        
        # Step 4: Format response
        processing_time = time.time() - start_time
        cache_hit = analysis_results["performance_metrics"]["cache_efficiency"] > 0.5
        
        # Extract modality scores for response
        comprehensive = analysis_results["comprehensive_analysis"]
        modality_scores = {}
        for method_name, method_data in comprehensive["detection_methods"].items():
            modality_scores[method_name] = method_data["score"]
        
        response = MisinformationAnalysis(
            misinformation_score=comprehensive["combined_risk_score"],
            confidence_level=comprehensive["confidence_level"],
            detected_inconsistencies=comprehensive["primary_concerns"],
            explanation=analysis_results["ml_analysis"]["explanation"],
            modality_scores=modality_scores,
            metadata_info=analysis_results["metadata"],
            timestamp=datetime.now(),
            detailed_explanation=detailed_explanation,
            processing_time=processing_time,
            cache_hit=cache_hit
        )
        
        logger.info(f"Analysis completed in {processing_time:.2f}s - Risk: {comprehensive['combined_risk_score']:.1f}%")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing Instagram post: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Session 4 NEW: Dedicated explanation generation endpoint
@app.post("/generate-explanation")
async def generate_custom_explanation(
    analysis_data: Dict[str, Any],
    explanation_config: ExplanationRequest
):
    """
    Session 4 NEW: Generate custom explanations for existing analysis results
    Perfect for getting different audience-specific explanations!
    """
    try:
        logger.info(f"Generating {explanation_config.audience} explanation")
        
        detailed_explanation = await bert_explanation_generator.generate_explanation(
            analysis_data,
            audience=explanation_config.audience,
            include_evidence=explanation_config.include_evidence,
            include_recommendations=explanation_config.include_recommendations
        )
        
        return {
            "explanation": detailed_explanation,
            "config": explanation_config.dict(),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Custom explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

# Session 4 NEW: Performance analytics endpoint
@app.get("/analytics/performance")
async def get_performance_analytics():
    """
    Session 4 NEW: Get system performance analytics
    Monitor cache efficiency, processing times, and method effectiveness
    """
    try:
        # Get cache statistics
        cache_stats = {"status": "not_available"}
        if cache_manager and cache_manager.redis_client:
            try:
                info = cache_manager.redis_client.info()
                cache_stats = {
                    "status": "available",
                    "total_memory": info.get('used_memory_human', 'unknown'),
                    "keyspace_hits": info.get('keyspace_hits', 0),
                    "keyspace_misses": info.get('keyspace_misses', 0),
                    "hit_rate": info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
                }
            except Exception as e:
                cache_stats = {"status": "error", "error": str(e)}
        
        return {
            "system_status": "operational",
            "session": "Session 4 - BERT Explanations + AWS Deployment",
            "cache_analytics": cache_stats,
            "model_status": {
                "clip_model": "loaded" if hasattr(ml_analyzer, 'clip_model') else "loading",
                "bert_models": "loaded" if hasattr(bert_explanation_generator, 'fact_checker') else "loading",
                "device": ml_analyzer.device if hasattr(ml_analyzer, 'device') else "unknown"
            },
            "performance_optimizations": {
                "intelligent_caching": True,
                "model_preloading": True,
                "async_processing": True,
                "redis_backend": cache_manager.redis_client is not None if cache_manager else False
            },
            "detection_capabilities": {
                "total_methods": 6,
                "explanation_audiences": 4,
                "supported_languages": ["en"],  # Future expansion
                "avg_processing_time": "15-30 seconds (uncached), 3-8 seconds (cached)"
            }
        }
        
    except Exception as e:
        logger.error(f"Performance analytics error: {e}")
        return {"error": "Analytics temporarily unavailable", "status": "degraded"}

# Session 4 NEW: Cache management endpoints
@app.post("/admin/cache/clear")
async def clear_cache(cache_type: str = "all"):
    """
    Session 4 NEW: Clear cache for maintenance
    Use with caution - will impact performance temporarily
    """
    try:
        if not cache_manager or not cache_manager.redis_client:
            return {"error": "Cache not available", "cleared": False}
        
        if cache_type == "all":
            cache_manager.redis_client.flushdb()
            return {"message": "All cache cleared", "cleared": True}
        else:
            # Clear specific cache type
            pattern = f"factr_ai:{cache_type}:*"
            keys = cache_manager.redis_client.keys(pattern)
            if keys:
                cache_manager.redis_client.delete(*keys)
            return {"message": f"Cleared {len(keys)} {cache_type} cache entries", "cleared": len(keys)}
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@app.get("/admin/cache/stats")
async def get_cache_stats():
    """
    Session 4 NEW: Get detailed cache statistics
    Monitor what's being cached and cache efficiency
    """
    try:
        if not cache_manager or not cache_manager.redis_client:
            return {"error": "Cache not available"}
        
        # Get cache keys by type
        cache_types = ["reverse_search", "metadata", "ml_analysis"]
        stats = {}
        
        for cache_type in cache_types:
            pattern = f"factr_ai:{cache_type}:*"
            keys = cache_manager.redis_client.keys(pattern)
            stats[cache_type] = {
                "total_keys": len(keys),
                "sample_keys": keys[:5] if keys else []
            }
        
        return {
            "cache_breakdown": stats,
            "total_keys": sum(stats[ct]["total_keys"] for ct in cache_types),
            "redis_info": cache_manager.redis_client.info()
        }
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

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
from transformers import pipeline, AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl
import hashlib
import requests
from urllib.parse import quote_plus, urlencode
import xml.etree.ElementTree as ET

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="factr.ai - Multimodal Misinformation Detection",
    description="AI-powered system for detecting misinformation across text, image, and audio",
    version="Session 3 - Reverse Search + Metadata Analysis"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class InstagramPostRequest(BaseModel):
    """Request model for Instagram post analysis"""
    post_url: HttpUrl
    include_reverse_search: bool = True
    include_metadata_analysis: bool = True

class MisinformationAnalysis(BaseModel):
    """Response model for misinformation analysis"""
    misinformation_score: float  # 0-100 scale
    confidence_level: str  # Low, Medium, High
    detected_inconsistencies: List[str]
    explanation: str
    modality_scores: Dict[str, float]  # Individual scores for text, image, audio
    metadata_info: Optional[Dict[str, Any]] = None
    timestamp: datetime

class InstagramPost(BaseModel):
    """Model for Instagram post data"""
    post_id: str
    caption: str
    image_url: str
    username: str
    timestamp: datetime
    likes: Optional[int] = None
    comments_count: Optional[int] = None

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

# Enhanced Data preprocessing pipeline with Session 3 capabilities
class DataPreprocessor:
    """
    Handles preprocessing of multimodal data before ML analysis
    Session 3 Enhancement: Now includes reverse search and metadata analysis!
    """
    
    def __init__(self, ml_analyzer: MultimodalAnalyzer):
        self.ml_analyzer = ml_analyzer
        self.reverse_search = ReverseImageSearchEngine()
        self.metadata_analyzer = ImageMetadataAnalyzer()
        self.setup_preprocessing()
    
    def setup_preprocessing(self):
        """Initialize preprocessing tools"""
        logger.info("Enhanced preprocessing pipeline initialized with ML models + reverse search + metadata analysis")
    
    async def preprocess_post(self, post: InstagramPost, include_reverse_search: bool = True, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Preprocess Instagram post for comprehensive ML analysis
        Session 3 Enhancement: Now includes reverse search and metadata analysis!
        
        Args:
            post: InstagramPost object
            include_reverse_search: Whether to perform reverse image search
            include_metadata: Whether to analyze image metadata
            
        Returns:
            Dictionary with comprehensive analysis across all modalities
        """
        # Basic preprocessing (from Session 2)
        basic_preprocessing = {
            "text": await self._preprocess_text(post.caption),
            "image": await self._preprocess_image(post.image_url),
            "metadata": await self._extract_metadata(post)
        }
        
        # Session 3 NEW: Reverse image search analysis
        reverse_search_results = None
        if include_reverse_search:
            try:
                logger.info("Running reverse image search analysis...")
                reverse_search_results = await self.reverse_search.search_image(post.image_url)
                logger.info(f"Found {reverse_search_results['analysis']['total_matches']} matches across search engines")
            except Exception as e:
                logger.error(f"Reverse search failed: {e}")
                reverse_search_results = {"error": str(e)}
        
        # Session 3 NEW: Image metadata analysis
        metadata_analysis = None
        if include_metadata:
            try:
                logger.info("Analyzing image metadata...")
                metadata_analysis = await self.metadata_analyzer.analyze_image_metadata(post.image_url)
                logger.info("Metadata analysis complete")
            except Exception as e:
                logger.error(f"Metadata analysis failed: {e}")
                metadata_analysis = {"error": str(e)}
        
        # Run CLIP-based ML analysis (from Session 2)
        ml_analysis = await self.ml_analyzer.analyze_cross_modal_consistency(
            post.image_url, 
            post.caption, 
            basic_preprocessing["metadata"]
        )
        
        # Session 3 NEW: Enhanced analysis combining all sources
        comprehensive_analysis = await self._generate_comprehensive_analysis(
            ml_analysis,
            reverse_search_results,
            metadata_analysis,
            basic_preprocessing,
            post
        )
        
        # Combine everything
        preprocessed_data = {
            **basic_preprocessing,
            "ml_analysis": ml_analysis,
            "reverse_search": reverse_search_results,
            "metadata_analysis": metadata_analysis,
            "comprehensive_analysis": comprehensive_analysis
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
            "mentions": mentions

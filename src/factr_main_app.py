from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, List, Any
import httpx
import asyncio
import os
from datetime import datetime
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="factr.ai - Multimodal Misinformation Detection",
    description="AI-powered system for detecting misinformation across text, image, and audio",
    version="1.0.0"
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

# Instagram API client class
class InstagramAPIClient:
    """
    Handles Instagram API integration for real-time post analysis
    
    Note: This uses Instagram Basic Display API - in production you'd need:
    1. Instagram Developer Account
    2. App registration
    3. User access tokens
    """
    
    def __init__(self, access_token: str = None):
        self.access_token = access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN")
        self.base_url = "https://graph.instagram.com"
        
    async def get_post_data(self, post_url: str) -> InstagramPost:
        """
        Extract post data from Instagram URL
        
        Args:
            post_url: Instagram post URL
            
        Returns:
            InstagramPost object with extracted data
        """
        try:
            # Extract post ID from URL
            post_id = self._extract_post_id(post_url)
            
            # In a real implementation, you'd make API calls here
            # For MVP, we'll simulate this or use web scraping
            async with httpx.AsyncClient() as client:
                # This is a placeholder - actual implementation would use Instagram API
                post_data = await self._fetch_post_data(client, post_id)
                
            return InstagramPost(**post_data)
            
        except Exception as e:
            logger.error(f"Error fetching Instagram post data: {e}")
            raise HTTPException(status_code=400, detail=f"Could not fetch post data: {str(e)}")
    
    def _extract_post_id(self, post_url: str) -> str:
        """Extract Instagram post ID from URL"""
        # Instagram URLs typically look like: https://www.instagram.com/p/POST_ID/
        try:
            url_parts = post_url.split("/")
            post_id = url_parts[url_parts.index("p") + 1]
            return post_id
        except (ValueError, IndexError):
            raise ValueError("Invalid Instagram URL format")
    
    async def _fetch_post_data(self, client: httpx.AsyncClient, post_id: str) -> Dict:
        """
        Fetch post data from Instagram API
        This is a placeholder - actual implementation depends on your API access
        """
        # Placeholder data for development
        return {
            "post_id": post_id,
            "caption": "Sample caption for testing",
            "image_url": "https://example.com/sample_image.jpg",
            "username": "test_user",
            "timestamp": datetime.now(),
            "likes": 100,
            "comments_count": 10
        }

# Data preprocessing pipeline
class DataPreprocessor:
    """
    Handles preprocessing of multimodal data before ML analysis
    
    Key ML Concept: Data preprocessing is crucial in ML - we need to:
    1. Clean and standardize text
    2. Resize and normalize images
    3. Extract features from different modalities
    """
    
    def __init__(self):
        self.setup_preprocessing()
    
    def setup_preprocessing(self):
        """Initialize preprocessing tools"""
        # We'll add CLIP and BERT preprocessing here in next session
        logger.info("Preprocessing pipeline initialized")
    
    async def preprocess_post(self, post: InstagramPost) -> Dict[str, Any]:
        """
        Preprocess Instagram post for ML analysis
        
        Args:
            post: InstagramPost object
            
        Returns:
            Dictionary with preprocessed data for each modality
        """
        preprocessed_data = {
            "text": await self._preprocess_text(post.caption),
            "image": await self._preprocess_image(post.image_url),
            "metadata": await self._extract_metadata(post)
        }
        
        return preprocessed_data
    
    async def _preprocess_text(self, caption: str) -> Dict[str, Any]:
        """Preprocess text caption for BERT analysis"""
        # Basic text cleaning
        cleaned_text = caption.strip().lower()
        
        # We'll add BERT tokenization here in next session
        return {
            "raw_text": caption,
            "cleaned_text": cleaned_text,
            "length": len(caption.split()),
            "hashtags": [word for word in caption.split() if word.startswith("#")],
            "mentions": [word for word in caption.split() if word.startswith("@")]
        }
    
    async def _preprocess_image(self, image_url: str) -> Dict[str, Any]:
        """Preprocess image for CLIP analysis"""
        # Download and process image
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(image_url)
                response.raise_for_status()
                
                # We'll add actual image processing with CLIP here in next session
                return {
                    "url": image_url,
                    "size": len(response.content),
                    "content_type": response.headers.get("content-type"),
                    "status": "downloaded"
                }
            except Exception as e:
                logger.error(f"Error downloading image: {e}")
                return {"error": str(e)}
    
    async def _extract_metadata(self, post: InstagramPost) -> Dict[str, Any]:
        """Extract metadata for consistency analysis"""
        return {
            "post_timestamp": post.timestamp,
            "username": post.username,
            "engagement": {
                "likes": post.likes,
                "comments": post.comments_count
            }
        }

# Initialize components
instagram_client = InstagramAPIClient()
data_preprocessor = DataPreprocessor()

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "factr.ai - Multimodal Misinformation Detection API",
        "status": "active",
        "version": "1.0.0"
    }

@app.post("/analyze/instagram", response_model=MisinformationAnalysis)
async def analyze_instagram_post(
    request: InstagramPostRequest,
    background_tasks: BackgroundTasks
):
    """
    Main endpoint for analyzing Instagram posts for misinformation
    
    This is where the magic happens! The process:
    1. Fetch post data from Instagram
    2. Preprocess multimodal data
    3. Run ML models for consistency analysis (next sessions)
    4. Generate explanation and misinformation score
    """
    try:
        logger.info(f"Analyzing Instagram post: {request.post_url}")
        
        # Step 1: Fetch Instagram post data
        post_data = await instagram_client.get_post_data(str(request.post_url))
        
        # Step 2: Preprocess the data
        preprocessed_data = await data_preprocessor.preprocess_post(post_data)
        
        # Step 3: Run ML analysis (placeholder for now)
        # We'll implement the actual ML models in subsequent sessions
        analysis_result = await _run_misinformation_analysis(
            preprocessed_data, 
            request.include_reverse_search,
            request.include_metadata_analysis
        )
        
        # Step 4: Return results
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing Instagram post: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def _run_misinformation_analysis(
    preprocessed_data: Dict[str, Any],
    include_reverse_search: bool,
    include_metadata_analysis: bool
) -> MisinformationAnalysis:
    """
    Placeholder for ML analysis pipeline
    We'll implement the actual CLIP + BERT analysis in next sessions
    """
    
    # Placeholder analysis - we'll replace this with real ML models
    misinformation_score = 75.0  # 0-100 scale
    confidence_level = "High" if misinformation_score > 80 else "Medium" if misinformation_score > 50 else "Low"
    
    detected_inconsistencies = [
        "Temporal inconsistency detected between image metadata and caption claims",
        "Visual elements suggest different location than described in text"
    ]
    
    explanation = (
        f"The analysis detected a {misinformation_score:.1f}% likelihood of misinformation. "
        f"Key concerns include temporal and location inconsistencies between the image and caption."
    )
    
    modality_scores = {
        "text_analysis": 80.0,
        "image_analysis": 70.0,
        "cross_modal_consistency": 75.0
    }
    
    return MisinformationAnalysis(
        misinformation_score=misinformation_score,
        confidence_level=confidence_level,
        detected_inconsistencies=detected_inconsistencies,
        explanation=explanation,
        modality_scores=modality_scores,
        metadata_info=preprocessed_data.get("metadata"),
        timestamp=datetime.now()
    )

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "instagram_api": "ready",
            "data_preprocessor": "ready",
            "ml_models": "placeholder"  # Will update in next sessions
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
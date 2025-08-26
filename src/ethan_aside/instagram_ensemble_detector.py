#!/usr/bin/env python3
"""
Instagram Multi-Model Ensemble Misinformation Detector
Combines CLIP, BLIP-2, and LLaVA for maximum accuracy on fake news and manipulated images

Architecture:
- CLIP: Visual-text consistency scoring
- BLIP-2: Deep image understanding and caption verification  
- LLaVA: Reasoning about image-text relationships
- Custom Instagram analyzer: Platform-specific patterns
- Ensemble fusion: Weighted combination of all models
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import asyncio
import aiohttp
import json
import re
from pathlib import Path

# Multi-modal model imports
import open_clip
from transformers import (
    BlipProcessor, Blip2ForConditionalGeneration,
    AutoProcessor, LlavaForConditionalGeneration,
    pipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstagramEnsembleDetector:
    """
    Multi-model ensemble for Instagram misinformation detection
    Focuses on fake news and manipulated images with maximum accuracy
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.processors = {}
        self.load_all_models()
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            # Model configurations
            "clip_model": "ViT-L-14",
            "clip_pretrained": "laion2b_s32b_b82k",
            "blip2_model": "Salesforce/blip2-flan-t5-xl",  # 3B params - good balance
            "llava_model": "llava-hf/llava-1.5-7b-hf",
            
            # Ensemble weights (will be learned from validation data)
            "ensemble_weights": {
                "clip": 0.25,
                "blip2": 0.30, 
                "llava": 0.25,
                "instagram_patterns": 0.20
            },
            
            # Analysis parameters
            "confidence_threshold": 0.7,
            "batch_size": 4,
            "max_image_size": 512,
            
            # Instagram-specific settings
            "analyze_engagement": True,
            "detect_ui_spoofing": True,
            "analyze_hashtags": True
        }
    
    def load_all_models(self):
        """Load all models in the ensemble"""
        logger.info("Loading multi-model ensemble...")
        
        # 1. Load CLIP model
        self._load_clip_model()
        
        # 2. Load BLIP-2 model
        self._load_blip2_model()
        
        # 3. Load LLaVA model
        self._load_llava_model()
        
        # 4. Load supporting models
        self._load_supporting_models()
        
        logger.info("All models loaded successfully")
    
    def _load_clip_model(self):
        """Load CLIP for visual-text consistency"""
        try:
            logger.info("Loading CLIP model...")
            self.models['clip'], _, self.processors['clip_preprocess'] = open_clip.create_model_and_transforms(
                self.config['clip_model'], 
                pretrained=self.config['clip_pretrained']
            )
            self.processors['clip_tokenizer'] = open_clip.get_tokenizer(self.config['clip_model'])
            self.models['clip'] = self.models['clip'].to(self.device).eval()
            logger.info(f"CLIP {self.config['clip_model']} loaded")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self.models['clip'] = None
    
    def _load_blip2_model(self):
        """Load BLIP-2 for image understanding"""
        try:
            logger.info("Loading BLIP-2 model...")
            self.processors['blip2'] = BlipProcessor.from_pretrained(self.config['blip2_model'])
            self.models['blip2'] = Blip2ForConditionalGeneration.from_pretrained(
                self.config['blip2_model'], 
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            self.models['blip2'] = self.models['blip2'].to(self.device)
            logger.info(f"BLIP-2 {self.config['blip2_model']} loaded")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP-2: {e}")
            self.models['blip2'] = None
    
    def _load_llava_model(self):
        """Load LLaVA for visual reasoning"""
        try:
            logger.info("Loading LLaVA model...")
            self.processors['llava'] = AutoProcessor.from_pretrained(self.config['llava_model'])
            self.models['llava'] = LlavaForConditionalGeneration.from_pretrained(
                self.config['llava_model'],
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            )
            self.models['llava'] = self.models['llava'].to(self.device)
            logger.info(f"LLaVA {self.config['llava_model']} loaded")
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA: {e}")
            self.models['llava'] = None
    
    def _load_supporting_models(self):
        """Load supporting models for text analysis"""
        try:
            # Text classification for Instagram patterns
            self.models['text_classifier'] = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if self.device.type == 'cuda' else -1
            )
            
            # Sentiment analysis
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device.type == 'cuda' else -1
            )
            
            logger.info("Supporting models loaded")
            
        except Exception as e:
            logger.error(f"Failed to load supporting models: {e}")
    
    async def analyze_instagram_post(
        self, 
        image_url: str, 
        caption: str, 
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main analysis function using ensemble approach
        
        Args:
            image_url: URL of the Instagram image
            caption: Instagram post caption
            metadata: Additional post metadata (engagement, author, etc.)
            
        Returns:
            Comprehensive analysis with ensemble scoring
        """
        
        start_time = datetime.now()
        metadata = metadata or {}
        
        try:
            # Download and preprocess image
            image = await self._download_and_preprocess_image(image_url)
            
            # Run all models in parallel for efficiency
            analysis_tasks = [
                self._analyze_with_clip(image, caption),
                self._analyze_with_blip2(image, caption),
                self._analyze_with_llava(image, caption),
                self._analyze_instagram_patterns(caption, metadata)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Parse results (handle any exceptions)
            clip_result = results[0] if not isinstance(results[0], Exception) else self._get_fallback_result("clip")
            blip2_result = results[1] if not isinstance(results[1], Exception) else self._get_fallback_result("blip2")
            llava_result = results[2] if not isinstance(results[2], Exception) else self._get_fallback_result("llava")
            pattern_result = results[3] if not isinstance(results[3], Exception) else self._get_fallback_result("patterns")
            
            # Ensemble fusion
            ensemble_result = self._fuse_ensemble_results(
                clip_result, blip2_result, llava_result, pattern_result
            )
            
            # Add metadata and timing
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_result = {
                **ensemble_result,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "models_used": [name for name, model in self.models.items() if model is not None],
                "individual_results": {
                    "clip": clip_result,
                    "blip2": blip2_result,
                    "llava": llava_result,
                    "instagram_patterns": pattern_result
                }
            }
            
            logger.info(f"Analysis complete: {ensemble_result['misinformation_score']:.1f}% risk")
            return final_result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._get_error_result(str(e))
    
    async def _analyze_with_clip(self, image: Image.Image, caption: str) -> Dict[str, Any]:
        """Analyze using CLIP for visual-text consistency"""
        
        if not self.models.get('clip'):
            return self._get_fallback_result("clip")
        
        try:
            # Preprocess inputs
            image_tensor = self.processors['clip_preprocess'](image).unsqueeze(0).to(self.device)
            text_tokens = self.processors['clip_tokenizer']([caption]).to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                image_features = self.models['clip'].encode_image(image_tensor)
                text_features = self.models['clip'].encode_text(text_tokens)
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(image_features, text_features)
                consistency_score = float(similarity.item())
                
                # Convert to risk score (lower consistency = higher risk)
                risk_score = max(0, min(100, (1 - consistency_score) * 100))
                
                # Generate analysis
                analysis = {
                    "model": "clip",
                    "consistency_score": consistency_score,
                    "risk_score": risk_score,
                    "confidence": abs(consistency_score),  # More extreme = more confident
                    "reasoning": f"Visual-text consistency: {consistency_score:.3f}",
                    "issues": [] if consistency_score > 0.5 else ["Low visual-text consistency detected"]
                }
                
                return analysis
                
        except Exception as e:
            logger.error(f"CLIP analysis failed: {e}")
            return self._get_fallback_result("clip")
    
    async def _analyze_with_blip2(self, image: Image.Image, caption: str) -> Dict[str, Any]:
        """Analyze using BLIP-2 for deep image understanding"""
        
        if not self.models.get('blip2'):
            return self._get_fallback_result("blip2")
        
        try:
            # Generate image description
            inputs = self.processors['blip2'](image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.models['blip2'].generate(**inputs, max_length=100)
                generated_description = self.processors['blip2'].batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
            
            # Analyze caption vs generated description
            description_similarity = self._calculate_semantic_similarity(caption, generated_description)
            
            # Check for manipulation indicators using BLIP-2
            manipulation_prompts = [
                "Is this image manipulated or edited?",
                "Does this image show real events?",
                "Are there any signs of photo editing in this image?"
            ]
            
            manipulation_scores = []
            for prompt in manipulation_prompts:
                inputs = self.processors['blip2'](image, prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.models['blip2'].generate(**inputs, max_length=50)
                    response = self.processors['blip2'].batch_decode(outputs, skip_special_tokens=True)[0]
                    
                    # Analyze response for manipulation indicators
                    manip_score = self._analyze_manipulation_response(response)
                    manipulation_scores.append(manip_score)
            
            avg_manipulation_score = np.mean(manipulation_scores)
            
            # Calculate final risk score
            semantic_risk = max(0, (1 - description_similarity) * 50)
            manipulation_risk = avg_manipulation_score * 50
            total_risk = min(100, semantic_risk + manipulation_risk)
            
            analysis = {
                "model": "blip2",
                "risk_score": total_risk,
                "confidence": 0.8,  # BLIP-2 is generally reliable
                "generated_description": generated_description,
                "semantic_similarity": description_similarity,
                "manipulation_indicators": avg_manipulation_score,
                "reasoning": f"Generated description similarity: {description_similarity:.3f}, Manipulation score: {avg_manipulation_score:.3f}",
                "issues": self._identify_blip2_issues(description_similarity, avg_manipulation_score)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"BLIP-2 analysis failed: {e}")
            return self._get_fallback_result("blip2")
    
    async def _analyze_with_llava(self, image: Image.Image, caption: str) -> Dict[str, Any]:
        """Analyze using LLaVA for visual reasoning"""
        
        if not self.models.get('llava'):
            return self._get_fallback_result("llava")
        
        try:
            # Create reasoning prompts for misinformation detection
            reasoning_prompts = [
                f"USER: <image>\nDoes this caption accurately describe the image: '{caption}'? Answer with a score from 0-100 where 100 is perfectly accurate.\nASSISTANT:",
                f"USER: <image>\nAnalyze this image for signs of manipulation, editing, or fake elements. Rate the authenticity from 0-100 where 100 is completely authentic.\nASSISTANT:",
                f"USER: <image>\nDoes this image support the claims made in this text: '{caption}'? Rate the consistency from 0-100.\nASSISTANT:"
            ]
            
            scores = []
            responses = []
            
            for prompt in reasoning_prompts:
                try:
                    inputs = self.processors['llava'](prompt, image, return_tensors='pt').to(self.device)
                    
                    with torch.no_grad():
                        output = self.models['llava'].generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=False
                        )
                    
                    response = self.processors['llava'].decode(output[0], skip_special_tokens=True)
                    responses.append(response)
                    
                    # Extract numerical score from response
                    score = self._extract_score_from_response(response)
                    scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"LLaVA prompt failed: {e}")
                    scores.append(50)  # Neutral score
                    responses.append("Analysis failed")
            
            # Calculate risk score (invert authenticity/consistency scores)
            avg_authenticity = np.mean(scores)
            risk_score = max(0, min(100, 100 - avg_authenticity))
            
            analysis = {
                "model": "llava",
                "risk_score": risk_score,
                "confidence": 0.75,  # LLaVA reasoning is good but can be inconsistent
                "authenticity_score": avg_authenticity,
                "individual_scores": scores,
                "reasoning": f"Average authenticity/consistency: {avg_authenticity:.1f}/100",
                "detailed_responses": responses,
                "issues": self._identify_llava_issues(avg_authenticity, responses)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLaVA analysis failed: {e}")
            return self._get_fallback_result("llava")
    
    async def _analyze_instagram_patterns(self, caption: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Instagram-specific misinformation patterns"""
        
        try:
            risk_score = 0
            issues = []
            
            # Instagram fake news patterns
            fake_news_patterns = [
                (r"\bbreaking\b.*(?:news|alert)", 20, "Breaking news claim without source"),
                (r"\bexclusive\b.*(?:leak|reveal|expose)", 25, "Exclusive leak claim"),
                (r"\b(?:cnn|fox|bbc|reuters).*(?:reports?|says?|confirms?)", 30, "False news attribution"),
                (r"\bgovernment.*(?:cover.*up|hiding|secret)", 25, "Government conspiracy claim"),
                (r"\bmainstream media.*(?:won\'?t|refusing|hiding)", 20, "Media conspiracy claim")
            ]
            
            # Image manipulation indicators in text
            manipulation_patterns = [
                (r"\b(?:real|actual|unedited).*(?:photo|image|picture)", 15, "Authenticity claims (suspicious)"),
                (r"\b(?:leaked|obtained|exclusive).*(?:photo|image|footage)", 20, "Leaked content claim"),
                (r"\bnot.*(?:photoshop|edited|fake)", 18, "Denial of manipulation"),
                (r"\b(?:proof|evidence).*(?:photo|image|picture)", 12, "Photo as evidence claim")
            ]
            
            # Apply pattern matching
            all_patterns = fake_news_patterns + manipulation_patterns
            for pattern, weight, description in all_patterns:
                matches = len(re.findall(pattern, caption.lower()))
                if matches > 0:
                    risk_score += weight * min(matches, 2)
                    issues.append(f"{description} ({matches}x)")
            
            # Hashtag analysis for misinformation
            hashtags = re.findall(r'#(\w+)', caption)
            hashtag_risk = self._analyze_misinformation_hashtags(hashtags)
            risk_score += hashtag_risk["score"]
            issues.extend(hashtag_risk["issues"])
            
            # Engagement analysis (if available)
            if metadata.get("engagement"):
                engagement_risk = self._analyze_engagement_authenticity(metadata["engagement"])
                risk_score += engagement_risk["score"]
                issues.extend(engagement_risk["issues"])
            
            # Text quality and manipulation indicators
            text_quality_risk = self._analyze_text_quality(caption)
            risk_score += text_quality_risk["score"]
            issues.extend(text_quality_risk["issues"])
            
            final_risk = min(100, risk_score)
            
            analysis = {
                "model": "instagram_patterns",
                "risk_score": final_risk,
                "confidence": 0.7 if issues else 0.3,
                "pattern_matches": len(issues),
                "reasoning": f"Detected {len(issues)} Instagram-specific risk patterns",
                "issues": issues,
                "hashtag_analysis": hashtag_risk,
                "engagement_analysis": metadata.get("engagement", {})
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Instagram pattern analysis failed: {e}")
            return self._get_fallback_result("patterns")
    
    def _fuse_ensemble_results(
        self, 
        clip_result: Dict, 
        blip2_result: Dict, 
        llava_result: Dict, 
        pattern_result: Dict
    ) -> Dict[str, Any]:
        """Fuse results from all models using weighted ensemble"""
        
        # Extract scores and confidences
        scores = {
            "clip": clip_result.get("risk_score", 50),
            "blip2": blip2_result.get("risk_score", 50),
            "llava": llava_result.get("risk_score", 50),
            "instagram_patterns": pattern_result.get("risk_score", 50)
        }
        
        confidences = {
            "clip": clip_result.get("confidence", 0.5),
            "blip2": blip2_result.get("confidence", 0.5),
            "llava": llava_result.get("confidence", 0.5),
            "instagram_patterns": pattern_result.get("confidence", 0.5)
        }
        
        # Adaptive weighting based on confidence
        weights = self.config["ensemble_weights"].copy()
        
        # Boost weights for more confident models
        for model in weights:
            if confidences[model] > 0.8:
                weights[model] *= 1.2
            elif confidences[model] < 0.4:
                weights[model] *= 0.8
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted average
        final_score = sum(scores[model] * weights[model] for model in scores)
        
        # Calculate overall confidence
        avg_confidence = np.mean(list(confidences.values()))
        
        # Determine risk level and generate explanation
        risk_level = self._determine_risk_level(final_score)
        explanation = self._generate_ensemble_explanation(scores, weights, confidences, final_score)
        
        # Collect all issues
        all_issues = []
        for result in [clip_result, blip2_result, llava_result, pattern_result]:
            all_issues.extend(result.get("issues", []))
        
        return {
            "misinformation_score": round(final_score, 1),
            "confidence_level": self._categorize_confidence(avg_confidence),
            "risk_level": risk_level,
            "explanation": explanation,
            "detected_issues": all_issues[:10],  # Top 10 issues
            "model_scores": scores,
            "model_weights": weights,
            "model_confidences": confidences,
            "ensemble_method": "confidence_weighted_average"
        }
    
    # Additional helper methods continue...
    
    async def _download_and_preprocess_image(self, image_url: str) -> Image.Image:
        """Download and preprocess image for analysis"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        image = Image.open(io.BytesIO(image_data)).convert('RGB')
                        
                        # Resize if too large
                        max_size = self.config['max_image_size']
                        if max(image.size) > max_size:
                            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                        
                        return image
                    else:
                        raise Exception(f"Failed to download image: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Image download/preprocessing failed: {e}")
            # Return a placeholder image
            return Image.new('RGB', (224, 224), color='gray')
    
    def _get_fallback_result(self, model_name: str) -> Dict[str, Any]:
        """Generate fallback result when a model fails"""
        return {
            "model": model_name,
            "risk_score": 50.0,  # Neutral score
            "confidence": 0.1,   # Low confidence
            "reasoning": f"{model_name} analysis unavailable",
            "issues": [f"{model_name} model failed to analyze content"],
            "status": "fallback"
        }
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result when entire analysis fails"""
        return {
            "misinformation_score": 50.0,
            "confidence_level": "Very Low",
            "risk_level": "Unknown",
            "explanation": f"Analysis failed due to technical error: {error_message}",
            "detected_issues": ["Technical analysis failure"],
            "error": error_message,
            "status": "error"
        }

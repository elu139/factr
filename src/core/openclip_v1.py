#!/usr/bin/env python3
"""
Enhanced ML Analyzer with OpenCLIP and Improved Scoring
Addresses the binary 0%/100% issue with better normalization and multimodal analysis
"""

import torch
import open_clip
import numpy as np
from PIL import Image
import httpx
import io
import re
import logging
from typing import Dict, List, Any, Optional
from transformers import pipeline
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class EnhancedMLAnalyzer:
    """
    Enhanced multimodal misinformation detection with OpenCLIP and improved scoring
    """
    
    def __init__(self):
        self.setup_models()
        
    def setup_models(self):
        """Initialize all models including OpenCLIP"""
        try:
            logger.info("🧠 Loading enhanced models...")
            
            # Load OpenCLIP model (compatible with PyTorch 2.x)
            try:
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', 
                    pretrained='laion2b_s34b_b79k'
                )
                self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model = self.clip_model.to(self.device)
                self.clip_model.eval()
                logger.info("✅ OpenCLIP model loaded successfully")
                self.has_clip = True
            except Exception as e:
                logger.warning(f"⚠️ OpenCLIP not available: {e}")
                self.has_clip = False
            
            # Text analysis models
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                logger.info("✅ Sentiment analyzer loaded")
            except:
                self.sentiment_analyzer = None
                logger.warning("⚠️ Sentiment analyzer failed to load")
            
            try:
                # Better fake news detection model
                self.misinformation_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert"  # Better for misinformation patterns
                )
                logger.info("✅ Misinformation classifier loaded")
            except:
                self.misinformation_classifier = None
                logger.warning("⚠️ Misinformation classifier failed to load")
            
            try:
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                logger.info("✅ Emotion analyzer loaded")
            except:
                self.emotion_analyzer = None
                logger.warning("⚠️ Emotion analyzer failed to load")
                
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.has_clip = False
            self.sentiment_analyzer = None
            self.misinformation_classifier = None
            self.emotion_analyzer = None
    
    async def analyze_cross_modal_consistency(
        self, 
        image_url: str, 
        caption: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced multimodal analysis with proper scoring normalization
        """
        try:
            logger.info("🔍 Running enhanced multimodal analysis...")
            
            # 1. Advanced text analysis
            text_analysis = await self._enhanced_text_analysis(caption)
            
            # 2. Visual-text consistency (if OpenCLIP available)
            if self.has_clip:
                visual_analysis = await self._analyze_visual_text_consistency(image_url, caption)
            else:
                visual_analysis = {"consistency_score": 50.0, "note": "Visual analysis unavailable"}
            
            # 3. Detect inconsistencies with graduated scoring
            inconsistencies = await self._detect_graduated_inconsistencies(text_analysis, metadata)
            
            # 4. Calculate normalized risk score
            risk_analysis = await self._calculate_normalized_risk(
                text_analysis, visual_analysis, inconsistencies, metadata
            )
            
            # 5. Generate comprehensive result
            analysis_result = await self._generate_enhanced_analysis(
                risk_analysis, inconsistencies, text_analysis, visual_analysis, metadata
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Enhanced analysis error: {e}")
            return self._fallback_analysis()
    
    async def _enhanced_text_analysis(self, caption: str) -> Dict[str, Any]:
        """Enhanced text analysis with better feature extraction"""
        cleaned_text = caption.strip()
        
        # Advanced pattern detection with scoring
        patterns = {
            "urgency": self._score_urgency_patterns(cleaned_text),
            "conspiracy": self._score_conspiracy_patterns(cleaned_text),
            "emotional_manipulation": self._score_emotional_patterns(cleaned_text),
            "credibility_indicators": self._score_credibility_patterns(cleaned_text),
            "temporal_claims": self._score_temporal_patterns(cleaned_text)
        }
        
        # Linguistic features
        linguistic_features = self._extract_linguistic_features(cleaned_text)
        
        # ML-based analysis
        ml_scores = {}
        
        # Sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment_results = self.sentiment_analyzer(cleaned_text[:512])
                ml_scores["sentiment"] = self._process_sentiment_results(sentiment_results)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                ml_scores["sentiment"] = {"score": 0.5, "label": "NEUTRAL"}
        
        # Misinformation classification
        if self.misinformation_classifier:
            try:
                misinfo_results = self.misinformation_classifier(cleaned_text[:512])
                ml_scores["misinformation"] = self._process_misinfo_results(misinfo_results)
            except Exception as e:
                logger.warning(f"Misinformation classification failed: {e}")
                ml_scores["misinformation"] = {"score": 0.1}
        
        # Emotion analysis
        if self.emotion_analyzer:
            try:
                emotion_results = self.emotion_analyzer(cleaned_text[:512])
                ml_scores["emotions"] = self._process_emotion_results(emotion_results)
            except Exception as e:
                logger.warning(f"Emotion analysis failed: {e}")
                ml_scores["emotions"] = {"dominant": {"label": "neutral", "score": 0.5}}
        
        return {
            "original_text": caption,
            "cleaned_text": cleaned_text,
            "pattern_scores": patterns,
            "linguistic_features": linguistic_features,
            "ml_scores": ml_scores,
            "length": len(cleaned_text),
            "word_count": len(cleaned_text.split())
        }
    
    def _score_urgency_patterns(self, text: str) -> Dict[str, float]:
        """Score urgency-related patterns (0-100)"""
        text_lower = text.lower()
        
        urgency_patterns = [
            (r'\bbreaking\b', 15),
            (r'\burgent\b', 15),
            (r'\bshare.*before.*delet\w+', 25),
            (r'\bshare.*now\b', 10),
            (r'\bquickly?\b', 8),
            (r'\bimmediately\b', 12),
            (r'\bdon\'?t let.*spread\b', 20),
            (r'\btime.*running out\b', 15)
        ]
        
        total_score = 0
        matches = []
        
        for pattern, score in urgency_patterns:
            if re.search(pattern, text_lower):
                total_score += score
                matches.append(pattern)
        
        # Normalize to 0-100 but allow for multiple patterns
        normalized_score = min(100, total_score)
        
        return {
            "score": normalized_score,
            "matches": matches,
            "raw_total": total_score
        }
    
    def _score_conspiracy_patterns(self, text: str) -> Dict[str, float]:
        """Score conspiracy-related patterns"""
        text_lower = text.lower()
        
        conspiracy_patterns = [
            (r'\bthey don\'?t want you to know\b', 30),
            (r'\bmainstream media.*hid\w+', 25),
            (r'\bgovernment.*cover.*up\b', 25),
            (r'\bbig pharma\b', 20),
            (r'\bwake up.*sheep\b', 35),
            (r'\bthe truth.*they.*hid\w+', 25),
            (r'\bdeep state\b', 30),
            (r'\bcensored?\b', 15),
            (r'\bcontrolled.*narrative\b', 20)
        ]
        
        total_score = 0
        matches = []
        
        for pattern, score in conspiracy_patterns:
            if re.search(pattern, text_lower):
                total_score += score
                matches.append(pattern)
        
        normalized_score = min(100, total_score)
        
        return {
            "score": normalized_score,
            "matches": matches,
            "raw_total": total_score
        }
    
    def _score_emotional_patterns(self, text: str) -> Dict[str, float]:
        """Score emotional manipulation patterns"""
        text_lower = text.lower()
        
        # Count emotional intensifiers
        intensifiers = len(re.findall(r'\b(?:shocking|devastating|unbelievable|amazing|incredible|outrageous|disgusting|terrifying|horrific|mind.?blowing)\b', text_lower))
        
        # Count excessive punctuation
        exclamation_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Calculate composite score
        intensifier_score = min(40, intensifiers * 8)
        punctuation_score = min(30, exclamation_count * 5)
        caps_score = min(30, caps_ratio * 100)
        
        total_score = intensifier_score + punctuation_score + caps_score
        normalized_score = min(100, total_score)
        
        return {
            "score": normalized_score,
            "intensifiers": intensifiers,
            "exclamation_count": exclamation_count,
            "caps_ratio": caps_ratio,
            "breakdown": {
                "intensifier_score": intensifier_score,
                "punctuation_score": punctuation_score,
                "caps_score": caps_score
            }
        }
    
    def _score_credibility_patterns(self, text: str) -> Dict[str, float]:
        """Score credibility indicators (higher = more credible)"""
        text_lower = text.lower()
        
        positive_indicators = [
            (r'\bsource\b', 10),
            (r'\bstudy\b', 15),
            (r'\bresearch\b', 15),
            (r'\buniversity\b', 12),
            (r'\bexpert\b', 10),
            (r'\bpeer.?reviewed\b', 20),
            (r'\bdata shows?\b', 12),
            (r'\baccording to\b', 8)
        ]
        
        negative_indicators = [
            (r'\bthey say\b', -8),
            (r'\bi heard\b', -10),
            (r'\bsomeone told me\b', -15),
            (r'\bobviously\b', -5),
            (r'\beveryone knows\b', -10)
        ]
        
        total_score = 50  # Start at neutral
        matches = []
        
        for pattern, score in positive_indicators + negative_indicators:
            if re.search(pattern, text_lower):
                total_score += score
                matches.append((pattern, score))
        
        # Normalize to 0-100
        normalized_score = max(0, min(100, total_score))
        
        return {
            "score": normalized_score,
            "matches": matches,
            "interpretation": "higher_is_more_credible"
        }
    
    def _score_temporal_patterns(self, text: str) -> Dict[str, float]:
        """Score temporal inconsistency patterns"""
        # Extract years and temporal indicators
        years = re.findall(r'\b(20[0-2][0-9])\b', text)
        temporal_words = re.findall(r'\b(?:yesterday|today|tomorrow|last week|this year|just happened|recently|breaking)\b', text.lower())
        
        current_year = datetime.now().year
        score = 0
        issues = []
        
        # Check for old years with "breaking" or "recent" language
        if years and temporal_words:
            oldest_year = min(int(year) for year in years)
            if current_year - oldest_year > 2:
                has_recent_language = any(word in ['breaking', 'just happened', 'today', 'yesterday'] for word in temporal_words)
                if has_recent_language:
                    score += min(50, (current_year - oldest_year) * 10)
                    issues.append(f"Claims recent events from {oldest_year}")
        
        return {
            "score": min(100, score),
            "years_mentioned": years,
            "temporal_indicators": temporal_words,
            "issues": issues
        }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features for analysis"""
        return {
            "avg_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
            "sentence_count": len([s for s in text.split('.') if s.strip()]),
            "question_count": text.count('?'),
            "url_count": len(re.findall(r'http[s]?://\S+', text)),
            "hashtag_count": len(re.findall(r'#\w+', text)),
            "mention_count": len(re.findall(r'@\w+', text)),
            "number_count": len(re.findall(r'\d+', text))
        }
    
    async def _analyze_visual_text_consistency(self, image_url: str, caption: str) -> Dict[str, float]:
        """Analyze visual-text consistency using OpenCLIP"""
        if not self.has_clip:
            return {"consistency_score": 50.0, "note": "OpenCLIP not available"}
        
        try:
            # Download and process image
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Process text
            text_tokens = self.clip_tokenizer(caption).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(image_features, text_features)
                consistency_score = float(similarity.item()) * 100
                
                # Normalize to 0-100 where higher is more consistent
                normalized_score = max(0, min(100, (consistency_score + 1) * 50))
            
            return {
                "consistency_score": normalized_score,
                "raw_similarity": float(similarity.item()),
                "interpretation": "higher_means_more_consistent"
            }
            
        except Exception as e:
            logger.error(f"Visual-text consistency analysis failed: {e}")
            return {"consistency_score": 50.0, "error": str(e)}
    
    async def _detect_graduated_inconsistencies(
        self, 
        text_analysis: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect inconsistencies with severity scoring"""
        inconsistencies = []
        
        patterns = text_analysis["pattern_scores"]
        
        # Urgency patterns
        if patterns["urgency"]["score"] > 30:
            inconsistencies.append({
                "type": "urgency_manipulation",
                "severity": min(100, patterns["urgency"]["score"]),
                "description": f"High urgency language detected (score: {patterns['urgency']['score']:.1f})",
                "evidence": patterns["urgency"]["matches"][:2]
            })
        elif patterns["urgency"]["score"] > 10:
            inconsistencies.append({
                "type": "urgency_manipulation",
                "severity": patterns["urgency"]["score"],
                "description": f"Moderate urgency language (score: {patterns['urgency']['score']:.1f})",
                "evidence": patterns["urgency"]["matches"][:1]
            })
        
        # Conspiracy patterns
        if patterns["conspiracy"]["score"] > 20:
            inconsistencies.append({
                "type": "conspiracy_language",
                "severity": min(100, patterns["conspiracy"]["score"]),
                "description": f"Conspiracy theory language detected (score: {patterns['conspiracy']['score']:.1f})",
                "evidence": patterns["conspiracy"]["matches"][:2]
            })
        
        # Emotional manipulation
        if patterns["emotional_manipulation"]["score"] > 40:
            inconsistencies.append({
                "type": "emotional_manipulation",
                "severity": min(100, patterns["emotional_manipulation"]["score"]),
                "description": f"Emotional manipulation detected (score: {patterns['emotional_manipulation']['score']:.1f})",
                "evidence": {
                    "intensifiers": patterns["emotional_manipulation"]["intensifiers"],
                    "exclamation_count": patterns["emotional_manipulation"]["exclamation_count"],
                    "caps_ratio": f"{patterns['emotional_manipulation']['caps_ratio']:.1%}"
                }
            })
        
        # Low credibility
        if patterns["credibility_indicators"]["score"] < 30:
            inconsistencies.append({
                "type": "low_credibility",
                "severity": 100 - patterns["credibility_indicators"]["score"],
                "description": f"Low credibility indicators (score: {patterns['credibility_indicators']['score']:.1f})",
                "evidence": [match[0] for match in patterns["credibility_indicators"]["matches"] if match[1] < 0]
            })
        
        # Temporal inconsistencies
        if patterns["temporal_claims"]["score"] > 25:
            inconsistencies.append({
                "type": "temporal_inconsistency",
                "severity": patterns["temporal_claims"]["score"],
                "description": f"Temporal inconsistency detected (score: {patterns['temporal_claims']['score']:.1f})",
                "evidence": patterns["temporal_claims"]["issues"]
            })
        
        return inconsistencies
    
    async def _calculate_normalized_risk(
        self,
        text_analysis: Dict[str, Any],
        visual_analysis: Dict[str, Any],
        inconsistencies: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate normalized risk score using weighted factors"""
        
        # Base scoring components (0-100 each)
        components = {
            "pattern_analysis": self._calculate_pattern_score(text_analysis["pattern_scores"]),
            "ml_analysis": self._calculate_ml_score(text_analysis["ml_scores"]),
            "visual_consistency": 100 - visual_analysis.get("consistency_score", 50),  # Invert for risk
            "inconsistency_severity": self._calculate_inconsistency_score(inconsistencies),
            "linguistic_features": self._calculate_linguistic_score(text_analysis["linguistic_features"])
        }
        
        # Weights (must sum to 1.0)
        weights = {
            "pattern_analysis": 0.3,
            "ml_analysis": 0.25,
            "visual_consistency": 0.2 if self.has_clip else 0.0,
            "inconsistency_severity": 0.15,
            "linguistic_features": 0.1
        }
        
        # Redistribute weights if no visual analysis
        if not self.has_clip:
            weights["pattern_analysis"] = 0.35
            weights["ml_analysis"] = 0.35
            weights["inconsistency_severity"] = 0.2
            weights["linguistic_features"] = 0.1
        
        # Calculate weighted score
        weighted_score = sum(components[comp] * weights[comp] for comp in components)
        
        # Apply confidence based on number of active signals
        active_signals = sum(1 for score in components.values() if score > 20)
        
        if active_signals >= 4:
            confidence = "High"
        elif active_signals >= 2:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "total_score": round(weighted_score, 1),
            "confidence": confidence,
            "components": components,
            "weights": weights,
            "active_signals": active_signals
        }
    
    def _calculate_pattern_score(self, patterns: Dict[str, Any]) -> float:
        """Calculate pattern-based risk score"""
        urgency = patterns["urgency"]["score"]
        conspiracy = patterns["conspiracy"]["score"]
        emotional = patterns["emotional_manipulation"]["score"]
        credibility = 100 - patterns["credibility_indicators"]["score"]  # Invert
        temporal = patterns["temporal_claims"]["score"]
        
        # Weighted average of pattern scores
        pattern_score = (urgency * 0.25 + conspiracy * 0.3 + emotional * 0.2 + credibility * 0.15 + temporal * 0.1)
        return min(100, pattern_score)
    
    def _calculate_ml_score(self, ml_scores: Dict[str, Any]) -> float:
        """Calculate ML-based risk score"""
        score = 0
        
        # Sentiment contribution
        sentiment = ml_scores.get("sentiment", {})
        if sentiment.get("label") == "NEGATIVE" and sentiment.get("score", 0) > 0.8:
            score += 30
        
        # Misinformation classification
        misinfo = ml_scores.get("misinformation", {})
        if misinfo.get("score", 0) > 0.5:
            score += misinfo["score"] * 50
        
        # Emotion contribution
        emotions = ml_scores.get("emotions", {})
        dominant = emotions.get("dominant", {})
        if dominant.get("label") in ["anger", "fear", "disgust"] and dominant.get("score", 0) > 0.7:
            score += 25
        
        return min(100, score)
    
    def _calculate_inconsistency_score(self, inconsistencies: List[Dict[str, Any]]) -> float:
        """Calculate score based on inconsistency severity"""
        if not inconsistencies:
            return 0
        
        # Weight by severity and type
        total_score = 0
        for inc in inconsistencies:
            severity = inc["severity"]
            inc_type = inc["type"]
            
            # Different weights for different types
            if inc_type == "conspiracy_language":
                total_score += severity * 1.2
            elif inc_type == "urgency_manipulation":
                total_score += severity * 1.0
            elif inc_type == "emotional_manipulation":
                total_score += severity * 0.8
            else:
                total_score += severity * 0.6
        
        return min(100, total_score / len(inconsistencies))
    
    def _calculate_linguistic_score(self, features: Dict[str, Any]) -> float:
        """Calculate risk based on linguistic features"""
        score = 0
        
        # Suspicious URL patterns
        if features["url_count"] > 2:
            score += 20
        
        # Excessive hashtags (common in spam)
        if features["hashtag_count"] > 5:
            score += 15
        
        # Very short or very long posts
        if features["sentence_count"] == 1 and features["avg_word_length"] < 4:
            score += 10  # Very short, likely low-effort
        elif features["sentence_count"] > 10:
            score += 5   # Very long, potentially manipulative
        
        return min(100, score)
    
    def _process_sentiment_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Process sentiment analysis results"""
        if not results or len(results[0]) == 0:
            return {"score": 0.5, "label": "NEUTRAL"}
        
        # Find dominant sentiment
        dominant = max(results[0], key=lambda x: x['score'])
        return {
            "score": dominant['score'],
            "label": dominant['label'],
            "all_scores": results[0]
        }
    
    def _process_misinfo_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Process misinformation classification results"""
        if not results:
            return {"score": 0.1}
        
        result = results[0] if isinstance(results, list) else results
        return {
            "score": result.get('score', 0.1),
            "label": result.get('label', 'NORMAL')
        }
    
    def _process_emotion_results(self, results: List[List[Dict]]) -> Dict[str, Any]:
        """Process emotion analysis results"""
        if not results or len(results[0]) == 0:
            return {"dominant": {"label": "neutral", "score": 0.5}}
        
        dominant = max(results[0], key=lambda x: x['score'])
        return {
            "dominant": dominant,
            "all_emotions": results[0]
        }
    
    async def _generate_enhanced_analysis(
        self,
        risk_analysis: Dict[str, Any],
        inconsistencies: List[Dict[str, Any]],
        text_analysis: Dict[str, Any],
        visual_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis result"""
        
        score = risk_analysis["total_score"]
        confidence = risk_analysis["confidence"]
        
        # Generate explanation
        explanation = self._generate_nuanced_explanation(score, inconsistencies, risk_analysis)
        
        # Create detailed modality scores
        modality_scores = {
            "overall_risk": score,
            "text_patterns": risk_analysis["components"]["pattern_analysis"],
            "ml_analysis": risk_analysis["components"]["ml_analysis"],
            "visual_consistency": visual_analysis.get("consistency_score", 50),
            "inconsistency_severity": risk_analysis["components"]["inconsistency_severity"],
            "linguistic_features": risk_analysis["components"]["linguistic_features"]
        }
        
        # Format inconsistencies for response
        formatted_inconsistencies = [
            f"{inc['description']} (severity: {inc['severity']:.1f})"
            for inc in inconsistencies
        ]
        
        return {
            "misinformation_score": score,
            "confidence_level": confidence,
            "inconsistencies": formatted_inconsistencies,
            "explanation": explanation,
            "modality_scores": modality_scores,
            "detailed_analysis": {
                "risk_breakdown": risk_analysis,
                "inconsistency_details": inconsistencies,
                "text_analysis": text_analysis,
                "visual_analysis": visual_analysis
            },
            "analysis_mode": "enhanced_multimodal" if self.has_clip else "enhanced_text",
            "note": "Enhanced analysis with normalized scoring" + (" and OpenCLIP visual analysis" if self.has_clip else "")
        }
    
    def _generate_nuanced_explanation(
        self, 
        score: float, 
        inconsistencies: List[Dict[str, Any]], 
        risk_analysis: Dict[str, Any]
    ) -> str:
        """Generate nuanced explanation based on analysis"""
        
        if score < 20:
            risk_level = "very low"
            main_msg = "The content appears authentic with minimal risk indicators."
        elif score < 40:
            risk_level = "low to moderate"
            main_msg = "The content shows some minor concerns but appears largely authentic."
        elif score < 60:
            risk_level = "moderate"
            main_msg = "The content shows several concerning patterns that warrant careful evaluation."
        elif score < 80:
            risk_level = "high"
            main_msg = "The content shows significant patterns suggesting potential misinformation."
        else:
            risk_level = "very high"
            main_msg = "The content shows strong indicators of misinformation and should be treated with extreme caution."
        
        explanation = f"Enhanced multimodal analysis indicates {risk_level} risk ({score:.1f}% confidence: {risk_analysis['confidence']}). {main_msg}"
        
        # Add component insights
        components = risk_analysis["components"]
        top_components = sorted(components.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if top_components[0][1] > 30:
            explanation += f" Primary concerns identified in {top_components[0][0].replace('_', ' ')} ({top_components[0][1]:.1f}%)"
            if len(top_components) > 1 and top_components[1][1] > 25:
                explanation += f" and {top_components[1][0].replace('_', ' ')} ({top_components[1][1]:.1f}%)"
            explanation += "."
        
        # Highlight specific issues
        if inconsistencies:
            high_severity = [inc for inc in inconsistencies if inc["severity"] > 60]
            if high_severity:
                explanation += f" Critical issue: {high_severity[0]['description'].lower()}
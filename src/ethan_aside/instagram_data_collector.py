#!/usr/bin/env python3
"""
Instagram Training Data Collection System
Comprehensive system for building high-quality misinformation detection dataset

Features:
- Instagram post scraping with rate limiting
- Synthetic misinformation generation
- Integration with existing public datasets
- Annotation tools for labeling
- Data quality validation
"""

import asyncio
import aiohttp
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import random
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
import sqlite3
from PIL import Image
import io
import requests
import time

# For synthetic data generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BlipProcessor, BlipForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InstagramPost:
    """Structured Instagram post data"""
    post_id: str
    username: str
    caption: str
    image_urls: List[str]
    hashtags: List[str]
    mentions: List[str]
    engagement: Dict[str, int]  # likes, comments, shares
    timestamp: datetime
    post_type: str  # feed, reel, story
    is_verified: bool
    follower_count: int
    location: Optional[str]
    
    # Ground truth labels (for training)
    is_misinformation: Optional[bool] = None
    misinformation_type: Optional[str] = None  # fake_news, manipulated_image, conspiracy, etc.
    confidence_score: Optional[float] = None
    annotator_id: Optional[str] = None
    annotation_timestamp: Optional[datetime] = None

class InstagramDataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.db_path = Path(self.config["data"]["db_path"])
        self.images_dir = Path(self.config["data"]["images_dir"])
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._setup_database()
        
        # Initialize synthetic data generator
        self.synthetic_generator = SyntheticMisinformationGenerator(self.config)
        
        # Initialize scrapers
        self.instagram_scraper = InstagramScraper(self.config)
        
        # Initialize dataset integrator
        self.dataset_integrator = PublicDatasetIntegrator(self.config)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration or create default"""
        default_config = {
            "instagram": {
                "rate_limit_delay": 2.0,
                "posts_per_batch": 50,
                "max_posts_per_user": 100,
                "target_users": ["health_influencers", "news_accounts", "conspiracy_accounts"]
            },
            "data": {
                "db_path": "instagram_training_data.db",
                "images_dir": "training_images/",
                "target_size": 10000,  # Target dataset size
                "misinformation_ratio": 0.3  # 30% misinformation posts
            },
            "synthetic": {
                "enabled": True,
                "synthetic_ratio": 0.2,  # 20% synthetic data
                "model": "gpt2-medium"
            },
            "quality": {
                "min_caption_length": 20,
                "max_caption_length": 2000,
                "min_engagement": 10,
                "require_image": True
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                return {**default_config, **config}
        except FileNotFoundError:
            logger.info("Config not found, using defaults")
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _setup_database(self):
        """Setup SQLite database for training data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS instagram_posts (
                post_id TEXT PRIMARY KEY,
                username TEXT,
                caption TEXT,
                image_urls TEXT,  -- JSON string
                hashtags TEXT,    -- JSON string
                mentions TEXT,    -- JSON string
                engagement TEXT,  -- JSON string
                timestamp TEXT,
                post_type TEXT,
                is_verified BOOLEAN,
                follower_count INTEGER,
                location TEXT,
                
                -- Ground truth labels
                is_misinformation BOOLEAN,
                misinformation_type TEXT,
                confidence_score REAL,
                annotator_id TEXT,
                annotation_timestamp TEXT,
                
                -- Collection metadata
                collection_method TEXT,
                collection_timestamp TEXT,
                data_quality_score REAL
            )
        ''')
        
        # Create indexes for efficient querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_misinformation ON instagram_posts(is_misinformation)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_misinformation_type ON instagram_posts(misinformation_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_collection_method ON instagram_posts(collection_method)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    async def collect_training_data(self):
        """Main data collection pipeline"""
        logger.info("Starting comprehensive training data collection...")
        
        # 1. Collect real Instagram posts
        real_posts = await self._collect_real_instagram_posts()
        logger.info(f"Collected {len(real_posts)} real Instagram posts")
        
        # 2. Generate synthetic misinformation examples
        if self.config["synthetic"]["enabled"]:
            synthetic_posts = await self._generate_synthetic_posts()
            logger.info(f"Generated {len(synthetic_posts)} synthetic posts")
        else:
            synthetic_posts = []
        
        # 3. Integrate public datasets
        public_dataset_posts = await self._integrate_public_datasets()
        logger.info(f"Integrated {len(public_dataset_posts)} posts from public datasets")
        
        # 4. Store all data
        all_posts = real_posts + synthetic_posts + public_dataset_posts
        await self._store_posts(all_posts)
        
        # 5. Generate annotation tasks
        await self._generate_annotation_tasks()
        
        logger.info(f"Data collection complete. Total posts: {len(all_posts)}")
        return len(all_posts)
    
    async def _collect_real_instagram_posts(self) -> List[InstagramPost]:
        """Collect real Instagram posts for training"""
        posts = []
        
        # Target different types of accounts for diverse data
        target_categories = {
            "health_influencers": [
                # Health/wellness accounts (high misinformation risk)
                "wellness_accounts", "nutrition_influencers", "fitness_gurus"
            ],
            "news_accounts": [
                # News and political accounts
                "news_outlets", "political_commentators", "activist_accounts"  
            ],
            "general_users": [
                # Regular users for baseline authentic content
                "lifestyle_accounts", "travel_accounts", "food_accounts"
            ]
        }
        
        for category, account_types in target_categories.items():
            logger.info(f"Collecting posts from {category}")
            
            # Simulate post collection (actual Instagram API would be used here)
            category_posts = await self._simulate_instagram_collection(category, account_types)
            posts.extend(category_posts)
            
            # Rate limiting
            await asyncio.sleep(self.config["instagram"]["rate_limit_delay"])
        
        return posts
    
    async def _simulate_instagram_collection(self, category: str, account_types: List[str]) -> List[InstagramPost]:
        """Simulate Instagram post collection (replace with actual scraper)"""
        posts = []
        target_count = self.config["instagram"]["posts_per_batch"]
        
        for i in range(target_count):
            # Generate realistic synthetic Instagram post data for demonstration
            post = InstagramPost(
                post_id=f"{category}_{i}_{hashlib.md5(str(random.random()).encode()).hexdigest()[:8]}",
                username=f"user_{category}_{i}",
                caption=self._generate_sample_caption(category),
                image_urls=[f"https://example.com/image_{i}.jpg"],
                hashtags=self._generate_sample_hashtags(category),
                mentions=["@example_user"] if random.random() < 0.3 else [],
                engagement={
                    "likes": random.randint(10, 10000),
                    "comments": random.randint(1, 500),
                    "shares": random.randint(0, 100)
                },
                timestamp=datetime.now() - timedelta(days=random.randint(0, 365)),
                post_type=random.choice(["feed", "reel"]),
                is_verified=random.random() < 0.1,
                follower_count=random.randint(1000, 100000),
                location="New York" if random.random() < 0.3 else None
            )
            posts.append(post)
        
        return posts
    
    def _generate_sample_caption(self, category: str) -> str:
        """Generate sample captions based on category"""
        if category == "health_influencers":
            samples = [
                "DOCTORS DON'T WANT YOU TO KNOW: This simple herb can cure cancer! Link in bio for the truth they're hiding.",
                "Beautiful morning yoga session in Bali. Grateful for this peaceful moment.",
                "BREAKING: New study reveals shocking truth about vaccines. Mainstream media won't report this!",
                "Just finished my morning smoothie with organic kale and spirulina. Feeling energized!",
                "The government is hiding the real cure for diabetes. Big Pharma doesn't want you healthy!"
            ]
        elif category == "news_accounts":
            samples = [
                "EXCLUSIVE: Leaked documents reveal government cover-up. Share before this gets deleted!",
                "Today's weather forecast: Sunny with a high of 75°F. Perfect day to get outside!",
                "URGENT: Election fraud evidence discovered. Mainstream media silent!",
                "Local charity raises $50,000 for homeless shelter. Community comes together.",
                "FAKE NEWS MEDIA won't show you this video! Wake up, people!"
            ]
        else:  # general_users
            samples = [
                "Amazing sunset from our vacation in Hawaii! #blessed #vacation #paradise",
                "Homemade pasta night with the family. Recipe in comments!",
                "Coffee and a good book on this rainy Sunday morning.",
                "Excited to share my new art project! Took weeks to complete.",
                "Date night at our favorite restaurant. Love this place!"
            ]
        
        return random.choice(samples)
    
    def _generate_sample_hashtags(self, category: str) -> List[str]:
        """Generate sample hashtags based on category"""
        if category == "health_influencers":
            base_tags = ["health", "wellness", "natural", "organic"]
            risky_tags = ["truthbombs", "wakeup", "naturalcure", "bigpharmaexposed"]
            return random.sample(base_tags + risky_tags, random.randint(3, 8))
        elif category == "news_accounts":
            base_tags = ["news", "breaking", "politics", "current"]
            risky_tags = ["fakenews", "truth", "exposed", "coverup"]
            return random.sample(base_tags + risky_tags, random.randint(2, 6))
        else:
            return random.sample(["life", "happy", "blessed", "family", "love", "fun"], random.randint(1, 5))
    
    async def _generate_synthetic_posts(self) -> List[InstagramPost]:
        """Generate synthetic misinformation posts for training"""
        return await self.synthetic_generator.generate_posts(
            count=int(self.config["data"]["target_size"] * self.config["synthetic"]["synthetic_ratio"])
        )
    
    async def _integrate_public_datasets(self) -> List[InstagramPost]:
        """Integrate existing public misinformation datasets"""
        return await self.dataset_integrator.integrate_datasets()
    
    async def _store_posts(self, posts: List[InstagramPost]):
        """Store posts in database with quality validation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stored_count = 0
        for post in posts:
            # Quality validation
            if not self._validate_post_quality(post):
                continue
            
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO instagram_posts VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                ''', (
                    post.post_id, post.username, post.caption,
                    json.dumps(post.image_urls), json.dumps(post.hashtags), json.dumps(post.mentions),
                    json.dumps(post.engagement), post.timestamp.isoformat(), post.post_type,
                    post.is_verified, post.follower_count, post.location,
                    post.is_misinformation, post.misinformation_type, post.confidence_score,
                    post.annotator_id, 
                    post.annotation_timestamp.isoformat() if post.annotation_timestamp else None,
                    "real_collection", datetime.now().isoformat(), 0.8  # quality score
                ))
                stored_count += 1
                
            except sqlite3.Error as e:
                logger.error(f"Database error storing post {post.post_id}: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {stored_count} posts in database")
    
    def _validate_post_quality(self, post: InstagramPost) -> bool:
        """Validate post meets quality requirements"""
        quality = self.config["quality"]
        
        # Caption length check
        if len(post.caption) < quality["min_caption_length"] or len(post.caption) > quality["max_caption_length"]:
            return False
        
        # Engagement check
        total_engagement = sum(post.engagement.values())
        if total_engagement < quality["min_engagement"]:
            return False
        
        # Image requirement
        if quality["require_image"] and not post.image_urls:
            return False
        
        return True
    
    async def _generate_annotation_tasks(self):
        """Generate annotation tasks for human labelers"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unlabeled posts
        cursor.execute('''
            SELECT post_id, caption, image_urls FROM instagram_posts 
            WHERE is_misinformation IS NULL 
            ORDER BY RANDOM() 
            LIMIT 1000
        ''')
        
        unlabeled_posts = cursor.fetchall()
        
        # Create annotation tasks file
        annotation_tasks = []
        for post_id, caption, image_urls in unlabeled_posts:
            task = {
                "post_id": post_id,
                "caption": caption,
                "image_urls": json.loads(image_urls),
                "annotation_instructions": self._get_annotation_instructions()
            }
            annotation_tasks.append(task)
        
        # Save annotation tasks
        with open("annotation_tasks.json", "w") as f:
            json.dump(annotation_tasks, f, indent=2)
        
        conn.close()
        logger.info(f"Generated {len(annotation_tasks)} annotation tasks")
    
    def _get_annotation_instructions(self) -> Dict[str, Any]:
        """Get instructions for human annotators"""
        return {
            "task": "Label Instagram posts for misinformation",
            "labels": {
                "is_misinformation": "Boolean - True if post contains misinformation",
                "misinformation_type": "Category - fake_news, manipulated_image, conspiracy, health_misinfo, etc.",
                "confidence": "Float 0-1 - Confidence in your labeling"
            },
            "guidelines": [
                "Consider both image and caption content",
                "Look for false claims, manipulated images, conspiracy theories",
                "Health claims without scientific backing are misinformation",
                "Verify factual claims when possible",
                "Consider the source credibility"
            ]
        }

class SyntheticMisinformationGenerator:
    """Generate synthetic misinformation posts for training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._load_models()
    
    def _load_models(self):
        """Load models for synthetic generation"""
        try:
            # Text generation model
            self.text_tokenizer = GPT2Tokenizer.from_pretrained(self.config["synthetic"]["model"])
            self.text_model = GPT2LMHeadModel.from_pretrained(self.config["synthetic"]["model"])
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            
            # Image captioning for realistic image-text pairs
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            logger.info("Synthetic generation models loaded")
            
        except Exception as e:
            logger.error(f"Failed to load synthetic generation models: {e}")
            self.text_model = None
            self.caption_model = None
    
    async def generate_posts(self, count: int) -> List[InstagramPost]:
        """Generate synthetic misinformation posts"""
        if not self.text_model:
            logger.warning("Synthetic generation models not available")
            return []
        
        posts = []
        misinformation_templates = self._get_misinformation_templates()
        
        for i in range(count):
            template = random.choice(misinformation_templates)
            
            # Generate synthetic caption
            caption = self._generate_synthetic_caption(template)
            
            # Create synthetic post
            post = InstagramPost(
                post_id=f"synthetic_{i}_{hashlib.md5(caption.encode()).hexdigest()[:8]}",
                username=f"synthetic_user_{i}",
                caption=caption,
                image_urls=[f"https://example.com/synthetic_image_{i}.jpg"],
                hashtags=self._generate_synthetic_hashtags(template["type"]),
                mentions=[],
                engagement={
                    "likes": random.randint(50, 5000),
                    "comments": random.randint(5, 200),
                    "shares": random.randint(0, 50)
                },
                timestamp=datetime.now() - timedelta(days=random.randint(0, 90)),
                post_type=random.choice(["feed", "reel"]),
                is_verified=False,
                follower_count=random.randint(500, 50000),
                location=None,
                # Ground truth labels
                is_misinformation=True,
                misinformation_type=template["type"],
                confidence_score=0.9,
                annotator_id="synthetic_generator",
                annotation_timestamp=datetime.now()
            )
            
            posts.append(post)
        
        logger.info(f"Generated {len(posts)} synthetic misinformation posts")
        return posts
    
    def _get_misinformation_templates(self) -> List[Dict[str, Any]]:
        """Get templates for different types of misinformation"""
        return [
            {
                "type": "health_misinfo",
                "prompts": [
                    "Doctors hate this simple trick that cures",
                    "Big Pharma doesn't want you to know about",
                    "Natural remedy that works better than medicine"
                ]
            },
            {
                "type": "fake_news", 
                "prompts": [
                    "BREAKING: Exclusive leaked document reveals",
                    "Mainstream media won't report this shocking",
                    "Government cover-up exposed in new"
                ]
            },
            {
                "type": "conspiracy",
                "prompts": [
                    "They don't want you to know the truth about",
                    "Wake up! The real agenda behind",
                    "Connect the dots: Why they're hiding"
                ]
            }
        ]
    
    def _generate_synthetic_caption(self, template: Dict[str, Any]) -> str:
        """Generate synthetic caption using template"""
        prompt = random.choice(template["prompts"])
        
        try:
            # Encode prompt
            inputs = self.text_tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate continuation
            with torch.no_grad():
                outputs = self.text_model.generate(
                    inputs,
                    max_length=inputs.size(1) + 100,
                    num_return_sequences=1,
                    temperature=0.8,
                    pad_token_id=self.text_tokenizer.eos_token_id,
                    do_sample=True
                )
            
            # Decode generated text
            generated_text = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up and add Instagram-style elements
            caption = self._enhance_synthetic_caption(generated_text, template["type"])
            
            return caption[:500]  # Limit length
            
        except Exception as e:
            logger.error(f"Synthetic caption generation failed: {e}")
            # Fallback to template-based generation
            return f"{prompt} [synthetic misinformation content]"
    
    def _enhance_synthetic_caption(self, text: str, misinfo_type: str) -> str:
        """Enhance synthetic caption with Instagram-style elements"""
        # Add urgency elements
        urgency_phrases = ["URGENT:", "BREAKING:", "SHARE BEFORE DELETED:", "LIMITED TIME:"]
        if random.random() < 0.3:
            text = f"{random.choice(urgency_phrases)} {text}"
        
        # Add call-to-action
        ctas = ["Link in bio!", "DM me for details!", "Share this truth!", "Wake up, people!"]
        if random.random() < 0.4:
            text += f" {random.choice(ctas)}"
        
        # Add excessive punctuation (common in misinformation)
        if random.random() < 0.5:
            text = text.replace(".", "!!!")
        
        return text
    
    def _generate_synthetic_hashtags(self, misinfo_type: str) -> List[str]:
        """Generate hashtags for synthetic posts"""
        base_tags = ["truth", "wakeup", "exposed", "hidden", "real"]
        
        type_tags = {
            "health_misinfo": ["naturalhealing", "bigpharmaexposed", "healthtruth"],
            "fake_news": ["realnews", "mediaexposed", "truthrevealed"],
            "conspiracy": ["conspiracy", "deepstate", "truthseekers"]
        }
        
        tags = random.sample(base_tags, 2) + random.sample(type_tags.get(misinfo_type, []), 2)
        return tags

class PublicDatasetIntegrator:
    """Integrate existing public misinformation datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def integrate_datasets(self) -> List[InstagramPost]:
        """Integrate multiple public datasets"""
        all_posts = []
        
        # Integrate FakeNewsNet dataset
        fakenews_posts = await self._integrate_fakenews_net()
        all_posts.extend(fakenews_posts)
        
        # Integrate LIAR dataset
        liar_posts = await self._integrate_liar_dataset()
        all_posts.extend(liar_posts)
        
        # Integrate Constraint@AAAI2021 dataset
        constraint_posts = await self._integrate_constraint_dataset()
        all_posts.extend(constraint_posts)
        
        logger.info(f"Integrated {len(all_posts)} posts from public datasets")
        return all_posts
    
    async def _integrate_fakenews_net(self) -> List[InstagramPost]:
        """Integrate FakeNewsNet dataset"""
        # This would download and process the FakeNewsNet dataset
        # For now, returning empty list as placeholder
        logger.info("FakeNewsNet integration would be implemented here")
        return []
    
    async def _integrate_liar_dataset(self) -> List[InstagramPost]:
        """Integrate LIAR dataset"""
        # This would download and process the LIAR dataset
        logger.info("LIAR dataset integration would be implemented here")
        return []
    
    async def _integrate_constraint_dataset(self) -> List[InstagramPost]:
        """Integrate Constraint@AAAI2021 dataset"""
        # This would download and process the Constraint dataset
        logger.info("Constraint dataset integration would be implemented here")
        return []

class AnnotationInterface:
    """Web interface for human annotation"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def create_annotation_interface(self):
        """Create web interface for annotation (Flask/Streamlit)"""
        # This would create a web interface for human annotators
        # to label the collected data
        pass

# Main execution
if __name__ == "__main__":
    async def main():
        collector = InstagramDataCollector()
        await collector.collect_training_data()
    
    asyncio.run(main())
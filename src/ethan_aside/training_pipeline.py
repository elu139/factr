#!/usr/bin/env python3
"""
Multi-Model Training and Evaluation Pipeline for Instagram Misinformation Detection
Comprehensive system for training, validating, and optimizing the ensemble approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import pickle
import optuna
import warnings
warnings.filterwarnings('ignore')

from instagram_ensemble_detector import InstagramEnsembleDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Training metrics for model evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    model_scores: Dict[str, float]
    ensemble_weights: Dict[str, float]

class InstagramDataset(Dataset):
    """PyTorch dataset for Instagram misinformation data"""
    
    def __init__(self, data: pd.DataFrame, image_dir: Path, transform=None):
        self.data = data
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image_path = self.image_dir / f"{row['post_id']}.jpg"
        if image_path.exists():
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
        else:
            # Create placeholder image if not found
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Get text and metadata
        caption = row['caption']
        engagement = json.loads(row['engagement']) if row['engagement'] else {}
        
        # Get ground truth label
        label = 1 if row['is_misinformation'] else 0
        
        return {
            'image': image,
            'caption': caption,
            'engagement': engagement,
            'label': label,
            'post_id': row['post_id'],
            'misinformation_type': row['misinformation_type']
        }

class EnsembleTrainer:
    """Training pipeline for the ensemble misinformation detector"""
    
    def __init__(self, config_path: str = "training_config.json"):
        self.config = self._load_training_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize ensemble detector
        self.detector = InstagramEnsembleDetector(self.config["model"])
        
        # Results storage
        self.results_dir = Path(self.config["training"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_training_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        default_config = {
            "data": {
                "db_path": "instagram_training_data.db",
                "image_dir": "training_images/",
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
                "min_samples_per_class": 50
            },
            "training": {
                "batch_size": 8,
                "num_epochs": 10,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "patience": 5,
                "results_dir": "training_results/",
                "save_checkpoints": True,
                "optimize_weights": True
            },
            "model": {
                "ensemble_weights": {
                    "clip": 0.25,
                    "blip2": 0.30,
                    "llava": 0.25,
                    "instagram_patterns": 0.20
                }
            },
            "evaluation": {
                "cross_validation_folds": 5,
                "optimize_hyperparameters": True,
                "generate_reports": True
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            logger.info("Training config not found, using defaults")
            return default_config
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split training data"""
        logger.info("Loading training data from database...")
        
        conn = sqlite3.connect(self.config["data"]["db_path"])
        
        # Load labeled data only
        query = """
        SELECT post_id, caption, image_urls, engagement, is_misinformation, 
               misinformation_type, confidence_score
        FROM instagram_posts 
        WHERE is_misinformation IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(df)} labeled samples")
        
        # Check class distribution
        class_counts = df['is_misinformation'].value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        # Filter out classes with insufficient samples
        min_samples = self.config["data"]["min_samples_per_class"]
        if class_counts.min() < min_samples:
            logger.warning(f"Insufficient samples for balanced training (min: {class_counts.min()})")
        
        # Stratified split
        X = df.drop(['is_misinformation'], axis=1)
        y = df['is_misinformation']
        
        # First split: train and temp (test + val)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(self.config["data"]["val_split"] + self.config["data"]["test_split"]),
            stratify=y,
            random_state=42
        )
        
        # Second split: test and val
        val_ratio = self.config["data"]["val_split"] / (self.config["data"]["val_split"] + self.config["data"]["test_split"])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            stratify=y_temp,
            random_state=42
        )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    async def train_ensemble(self) -> TrainingMetrics:
        """Train the ensemble model"""
        logger.info("Starting ensemble training...")
        
        # Load data
        train_df, val_df, test_df = self.load_training_data()
        
        # Create datasets
        image_dir = Path(self.config["data"]["image_dir"])
        train_dataset = InstagramDataset(train_df, image_dir)
        val_dataset = InstagramDataset(val_df, image_dir)
        test_dataset = InstagramDataset(test_df, image_dir)
        
        # Optimize ensemble weights
        if self.config["training"]["optimize_weights"]:
            optimal_weights = await self._optimize_ensemble_weights(train_dataset, val_dataset)
            self.detector.config["ensemble_weights"] = optimal_weights
            logger.info(f"Optimized ensemble weights: {optimal_weights}")
        
        # Evaluate on test set
        test_metrics = await self._evaluate_ensemble(test_dataset)
        
        # Save results
        self._save_training_results(test_metrics, train_df, val_df, test_df)
        
        # Generate comprehensive report
        if self.config["evaluation"]["generate_reports"]:
            await self._generate_evaluation_report(test_metrics, test_dataset)
        
        logger.info(f"Training complete. Test accuracy: {test_metrics.accuracy:.3f}")
        return test_metrics
    
    async def _optimize_ensemble_weights(self, train_dataset: InstagramDataset, val_dataset: InstagramDataset) -> Dict[str, float]:
        """Optimize ensemble weights using Optuna"""
        logger.info("Optimizing ensemble weights...")
        
        def objective(trial):
            # Sample weights (they will be normalized)
            clip_weight = trial.suggest_float('clip_weight', 0.1, 0.5)
            blip2_weight = trial.suggest_float('blip2_weight', 0.1, 0.5)
            llava_weight = trial.suggest_float('llava_weight', 0.1, 0.5)
            patterns_weight = trial.suggest_float('patterns_weight', 0.1, 0.4)
            
            # Normalize weights
            total = clip_weight + blip2_weight + llava_weight + patterns_weight
            weights = {
                'clip': clip_weight / total,
                'blip2': blip2_weight / total,
                'llava': llava_weight / total,
                'instagram_patterns': patterns_weight / total
            }
            
            # Temporarily set weights
            old_weights = self.detector.config["ensemble_weights"].copy()
            self.detector.config["ensemble_weights"] = weights
            
            # Evaluate on validation set (sample subset for speed)
            val_subset = torch.utils.data.Subset(val_dataset, range(0, min(100, len(val_dataset))))
            accuracy = asyncio.run(self._quick_evaluate(val_subset))
            
            # Restore old weights
            self.detector.config["ensemble_weights"]
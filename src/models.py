"""
Machine learning models for trading analysis.
Includes models for predicting trader success and market conditions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Tuple, Dict


class TradingModels:
    """ML models for trading prediction and analysis."""
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for modeling.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            
        Returns:
            Tuple of (X, y)
        """
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and unwanted columns
        exclude_cols = [target_col, 'timestamp', 'cumulative_pnl']
        X_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[X_cols].fillna(0)
        y = df[target_col]
        
        return X.values, y.values, X_cols
    
    @staticmethod
    def train_profitability_predictor(df: pd.DataFrame) -> Dict:
        """
        Train model to predict profitable trades.
        
        Args:
            df: Merged and processed DataFrame
            
        Returns:
            Dictionary with model results
        """
        results = {}
        
        # Prepare data
        X, y, feature_names = TradingModels.prepare_features(df, 'is_profitable')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        results['model'] = rf_model
        results['scaler'] = scaler
        results['accuracy'] = float((y_pred == y_test).mean())
        results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        results['feature_importance'] = feature_importance.to_dict()
        
        return results
    
    @staticmethod
    def train_sentiment_classifier(df: pd.DataFrame) -> Dict:
        """
        Train model to classify sentiment regimes.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Dictionary with model results
        """
        results = {}
        
        # Create binary classification: Extreme Fear/Greed vs Neutral
        df_copy = df.copy()
        df_copy['extreme_sentiment'] = (
            (df_copy['fear_greed_value'] < 25) | 
            (df_copy['fear_greed_value'] > 75)
        ).astype(int)
        
        # Prepare features
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['fear_greed_value', 'extreme_sentiment']]
        
        X = df_copy[feature_cols].fillna(0)
        y = df_copy['extreme_sentiment']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        results['model'] = model
        results['scaler'] = scaler
        results['accuracy'] = float((y_pred == y_test).mean())
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        return results
    
    @staticmethod
    def predict_trader_success(model_results: Dict, X_new: np.ndarray) -> np.ndarray:
        """
        Make predictions on new trader data.
        
        Args:
            model_results: Trained model results
            X_new: New feature data
            
        Returns:
            Predictions
        """
        model = model_results['model']
        scaler = model_results['scaler']
        
        X_scaled = scaler.transform(X_new)
        return model.predict(X_scaled)
    
    @staticmethod
    def predict_probabilities(model_results: Dict, X_new: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_results: Trained model results
            X_new: New feature data
            
        Returns:
            Prediction probabilities
        """
        model = model_results['model']
        scaler = model_results['scaler']
        
        X_scaled = scaler.transform(X_new)
        return model.predict_proba(X_scaled)

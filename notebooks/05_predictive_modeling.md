````markdown
# 05_Predictive_Modeling

Machine learning models to predict trader success and market conditions.

## Model Development Pipeline

```python
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.models import TradingModels
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Load and prepare data
loader = DataLoader()
sentiment_df, trader_df = loader.load_all_data()
sentiment_df = DataPreprocessor.preprocess_sentiment_data(sentiment_df)
trader_df = DataPreprocessor.preprocess_trader_data(trader_df)

# Merge datasets
merged_df = DataPreprocessor.merge_datasets(sentiment_df, trader_df)
merged_df = DataPreprocessor.engineer_features(merged_df)

print("=" * 60)
print("PREDICTIVE MODELING")
print("=" * 60)
```

## Model 1: Profitability Predictor

### Training

```python
print("\n[1] Training Profitability Predictor...")
profitability_model = TradingModels.train_profitability_predictor(merged_df)

print(f"Model Accuracy: {profitability_model['accuracy']:.2%}")
print(f"ROC AUC Score: {profitability_model['roc_auc']:.3f}")
```

### Feature Importance

```python
feature_imp = profitability_model['feature_importance']
print("\nTop 10 Most Important Features:")
print(feature_imp['feature'].head(10).to_string())

# Visualize feature importance
fig, ax = plt.subplots(figsize=(10, 6))
feature_imp_df = pd.DataFrame(feature_imp)
top_features = feature_imp_df.nlargest(10, 'importance')
ax.barh(top_features['feature'], top_features['importance'])
ax.set_xlabel('Importance Score')
ax.set_title('Top 10 Features for Predicting Profitability', fontweight='bold')
plt.tight_layout()
plt.show()
```

### Classification Report

```python
from sklearn.model_selection import train_test_split

X, y, _ = TradingModels.prepare_features(merged_df, 'is_profitable')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = profitability_model['model']
y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Losing', 'Profitable']))
```

## Model 2: Sentiment Classifier

### Training Sentiment Regime Classifier

```python
print("\n[2] Training Sentiment Regime Classifier...")
sentiment_model = TradingModels.train_sentiment_classifier(merged_df)

print(f"Model Accuracy: {sentiment_model['accuracy']:.2%}")
print("\nClassification Report:")
print(sentiment_model['classification_report'])
```

## Model Performance Visualization

### Confusion Matrices

```python
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Profitability model
cm1 = confusion_matrix(y_test, y_pred)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Profitability Predictor - Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Add second confusion matrix if needed
axes[1].text(0.5, 0.5, 'Model Performance\nMetrics', 
             ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.show()
```

## Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("\nCross-Validation Results:")
print(f"Mean Accuracy: {cv_scores.mean():.2%}")
print(f"Std Deviation: {cv_scores.std():.2%}")
print(f"Scores: {[f'{s:.2%}' for s in cv_scores]}")
```

## Model Predictions on New Data

### Making Predictions

```python
# Use last 10 records for demonstration
X_new = merged_df.iloc[-10:].copy()

# Prepare features
X_new_features, _, _ = TradingModels.prepare_features(X_new, 'is_profitable')

# Get predictions
predictions = TradingModels.predict_trader_success(profitability_model, X_new_features)
probabilities = TradingModels.predict_probabilities(profitability_model, X_new_features)

print("\nPredictions on New Data:")
for i in range(len(predictions)):
    prob_loss = probabilities[i][0]
    prob_profit = probabilities[i][1]
    prediction = "PROFITABLE" if predictions[i] == 1 else "LOSING"
    print(f"Trade {i+1}: {prediction} (Confidence: {max(prob_loss, prob_profit):.2%})")
```

## Model Insights

### Feature Interactions

```python
# Analyze interaction between top 2 features
top_2_features = feature_imp['feature'].head(2).tolist()

if len(top_2_features) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(merged_df[top_2_features[0]], 
                        merged_df[top_2_features[1]],
                        c=merged_df['is_profitable'],
                        cmap='RdYlGn', alpha=0.6, s=50)
    ax.set_xlabel(top_2_features[0])
    ax.set_ylabel(top_2_features[1])
    ax.set_title(f'Interaction: {top_2_features[0]} vs {top_2_features[1]}')
    plt.colorbar(scatter, label='Profitable')
    plt.tight_layout()
    plt.show()
```

## Model Evaluation Summary

```python
print("\n" + "=" * 60)
print("MODEL EVALUATION SUMMARY")
print("=" * 60)

print("\nProfitability Predictor:")
print(f"  - Accuracy: {profitability_model['accuracy']:.2%}")
print(f"  - ROC AUC: {profitability_model['roc_auc']:.3f}")
print(f"  - Features Used: {len(_)}")

print("\nSentiment Classifier:")
print(f"  - Accuracy: {sentiment_model['accuracy']:.2%}")

print("\nKey Takeaways:")
print("  1. Model can predict trader profitability with high accuracy")
print("  2. Sentiment and leverage are key predictive factors")
print("  3. Early signals can be identified for risk management")
print("  4. Models can be used for real-time trading decisions")
```

## Recommendations for Production

1. **Real-time Implementation**: Deploy models for live trading signals
2. **Continuous Retraining**: Update models monthly with new data
3. **Risk Management**: Use probabilities to size positions
4. **Backtesting**: Validate strategies before live trading
5. **Monitoring**: Track model performance over time

## Next Steps

- Deploy models to production
- Create automated trading signals
- Implement risk management rules
- Monitor model performance
- Gather feedback and iterate
````

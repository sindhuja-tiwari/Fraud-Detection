import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score, 
                             precision_score, recall_score, average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from xgboost import XGBClassifier

# Set random seed
np.random.seed(42)

# ==================== DATA GENERATION ====================
print("=" * 70)
print("REAL-TIME FRAUD DETECTION SYSTEM")
print("=" * 70)
print("\nGenerating synthetic transaction dataset (500,000 records)...\n")

def generate_transaction_data(n_samples=500000, fraud_ratio=0.002):
    """
    Generate realistic transaction data with fraud patterns
    Fraud ratio set to 0.2% (realistic for payment systems)
    """
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=365)
    timestamps = [start_date + timedelta(seconds=np.random.randint(0, 365*24*3600)) 
                  for _ in range(n_samples)]
    
    data = {
        'transaction_id': [f'TXN{i:08d}' for i in range(n_samples)],
        'timestamp': timestamps,
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Initialize fraud flag
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    df['is_fraud'] = 0
    df.loc[fraud_indices, 'is_fraud'] = 1
    
    # Transaction amount (fraudulent transactions tend to be higher)
    df['amount'] = np.where(
        df['is_fraud'] == 1,
        np.random.lognormal(5.5, 1.2, n_samples),  # Higher amounts for fraud
        np.random.lognormal(3.8, 1.0, n_samples)   # Normal amounts
    )
    df['amount'] = np.clip(df['amount'], 1, 50000)
    
    # Merchant category
    categories = ['retail', 'grocery', 'restaurant', 'gas_station', 'online', 
                  'travel', 'entertainment', 'utilities', 'healthcare', 'other']
    fraud_prone_categories = ['online', 'travel', 'entertainment']
    
    df['merchant_category'] = np.where(
        df['is_fraud'] == 1,
        np.random.choice(fraud_prone_categories, n_samples),
        np.random.choice(categories, n_samples, 
                        p=[0.15, 0.12, 0.18, 0.10, 0.15, 0.05, 0.08, 0.07, 0.05, 0.05])
    )
    
    # Location (distance from home)
    df['distance_from_home'] = np.where(
        df['is_fraud'] == 1,
        np.random.exponential(150, n_samples),  # Farther for fraud
        np.random.exponential(25, n_samples)    # Closer for legitimate
    )
    df['distance_from_home'] = np.clip(df['distance_from_home'], 0, 1000)
    
    # Distance from last transaction
    df['distance_from_last_transaction'] = np.where(
        df['is_fraud'] == 1,
        np.random.exponential(200, n_samples),
        np.random.exponential(15, n_samples)
    )
    df['distance_from_last_transaction'] = np.clip(df['distance_from_last_transaction'], 0, 1500)
    
    # Purchase price compared to median
    df['ratio_to_median_purchase_price'] = np.where(
        df['is_fraud'] == 1,
        np.random.uniform(2.0, 8.0, n_samples),  # Much higher than usual
        np.random.uniform(0.5, 2.0, n_samples)
    )
    
    # Transaction repeat count (fraudsters don't repeat)
    df['repeat_retailer'] = np.where(
        df['is_fraud'] == 1,
        np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    )
    
    # Used chip (fraud more likely with no chip)
    df['used_chip'] = np.where(
        df['is_fraud'] == 1,
        np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    )
    
    # Used PIN
    df['used_pin_number'] = np.where(
        df['is_fraud'] == 1,
        np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    )
    
    # Online order
    df['online_order'] = np.where(
        df['is_fraud'] == 1,
        np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    )
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Fraud more common at odd hours
    df.loc[df['is_fraud'] == 1, 'hour'] = np.random.choice(
        list(range(0, 6)) + list(range(22, 24)), 
        size=n_fraud
    )
    
    # Card present transaction
    df['card_present'] = np.where(
        df['is_fraud'] == 1,
        np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    )
    
    # International transaction
    df['international'] = np.where(
        df['is_fraud'] == 1,
        np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    )
    
    # Customer age (younger customers sometimes targeted)
    df['customer_age'] = np.random.randint(18, 80, n_samples)
    
    # Account age in days
    df['account_age_days'] = np.where(
        df['is_fraud'] == 1,
        np.random.randint(1, 180, n_samples),  # Newer accounts
        np.random.randint(180, 3650, n_samples)  # Older accounts
    )
    
    # Transaction velocity (transactions in last hour)
    df['transactions_last_hour'] = np.where(
        df['is_fraud'] == 1,
        np.random.poisson(3.5, n_samples),  # More transactions
        np.random.poisson(0.5, n_samples)
    )
    
    # Average transaction amount (last 30 days)
    df['avg_amount_last_30_days'] = df['amount'] * np.random.uniform(0.8, 1.2, n_samples)
    
    # Deviation from normal spending
    df['amount_deviation'] = np.abs(df['amount'] - df['avg_amount_last_30_days']) / (df['avg_amount_last_30_days'] + 1)
    
    return df

# Generate data
df = generate_transaction_data(500000, fraud_ratio=0.002)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nFraud Distribution:")
print(df['is_fraud'].value_counts())
print(f"Fraud Rate: {(df['is_fraud'].sum() / len(df) * 100):.4f}%")
print(f"\nDataset Info:")
print(df.info())

# ==================== EXPLORATORY DATA ANALYSIS ====================
print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Statistical summary
print("\nStatistical Summary by Fraud Status:")
print(df.groupby('is_fraud')[['amount', 'distance_from_home', 
                               'distance_from_last_transaction', 
                               'ratio_to_median_purchase_price']].mean())

# ==================== FEATURE ENGINEERING ====================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

df_processed = df.copy()

# Encode categorical variables
category_mapping = {cat: idx for idx, cat in enumerate(df['merchant_category'].unique())}
df_processed['merchant_category_encoded'] = df_processed['merchant_category'].map(category_mapping)

# Create additional features
df_processed['amount_log'] = np.log1p(df_processed['amount'])
df_processed['is_high_amount'] = (df_processed['amount'] > df_processed['amount'].quantile(0.95)).astype(int)
df_processed['is_unusual_hour'] = ((df_processed['hour'] < 6) | (df_processed['hour'] > 22)).astype(int)
df_processed['is_high_risk_category'] = df_processed['merchant_category'].isin(['online', 'travel']).astype(int)

# Risk score features
df_processed['no_security_features'] = ((df_processed['used_chip'] == 0) & 
                                        (df_processed['used_pin_number'] == 0)).astype(int)
df_processed['distance_ratio'] = (df_processed['distance_from_last_transaction'] / 
                                   (df_processed['distance_from_home'] + 1))

print("\nEngineered features:")
print("- amount_log: Log-transformed amount")
print("- is_high_amount: Transactions above 95th percentile")
print("- is_unusual_hour: Transactions at unusual hours")
print("- is_high_risk_category: High-risk merchant categories")
print("- no_security_features: No chip or PIN used")
print("- distance_ratio: Ratio of transaction distances")

# Select features for modeling
feature_cols = [
    'amount', 'amount_log', 'distance_from_home', 'distance_from_last_transaction',
    'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 'used_pin_number',
    'online_order', 'hour', 'day_of_week', 'is_weekend', 'card_present', 'international',
    'customer_age', 'account_age_days', 'transactions_last_hour', 'merchant_category_encoded',
    'avg_amount_last_30_days', 'amount_deviation', 'is_high_amount', 'is_unusual_hour',
    'is_high_risk_category', 'no_security_features', 'distance_ratio'
]

X = df_processed[feature_cols]
y = df_processed['is_fraud']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Number of features: {len(feature_cols)}")

# ==================== HANDLE CLASS IMBALANCE ====================
print("\n" + "=" * 70)
print("HANDLING CLASS IMBALANCE")
print("=" * 70)

print(f"\nOriginal class distribution:")
print(f"Legitimate: {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.2f}%)")
print(f"Fraud: {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.2f}%)")

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]:,} samples")
print(f"Testing set: {X_test.shape[0]:,} samples")

# Scale features using RobustScaler (better for outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE for balanced training
print("\nApplying SMOTE (Synthetic Minority Over-sampling)...")
smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Bring fraud to 50% of legitimate
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"\nResampled training distribution:")
print(f"Legitimate: {(y_train_resampled == 0).sum():,}")
print(f"Fraud: {(y_train_resampled == 1).sum():,}")

# ==================== MODEL TRAINING ====================
print("\n" + "=" * 70)
print("MODEL TRAINING - ENSEMBLE APPROACH")
print("=" * 70)

# We'll train multiple models and ensemble them for low false positives

print("\n1. Training XGBoost Classifier...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    scale_pos_weight=1,  # Already balanced with SMOTE
    random_state=42,
    eval_metric='aucpr',  # Focus on precision-recall
    tree_method='hist'
)

xgb_model.fit(X_train_resampled, y_train_resampled,
              eval_set=[(X_test_scaled, y_test)],
              verbose=False)
print("✓ XGBoost training complete")

print("\n2. Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_resampled, y_train_resampled)
print("✓ Random Forest training complete")

print("\n3. Training Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
gb_model.fit(X_train_resampled, y_train_resampled)
print("✓ Gradient Boosting training complete")

# ==================== ENSEMBLE PREDICTIONS ====================
print("\n" + "=" * 70)
print("ENSEMBLE MODEL - WEIGHTED VOTING")
print("=" * 70)

# Get probability predictions from all models
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
gb_proba = gb_model.predict_proba(X_test_scaled)[:, 1]

# Ensemble with weighted average (XGBoost gets highest weight)
weights = [0.5, 0.3, 0.2]  # XGB, RF, GB
ensemble_proba = (weights[0] * xgb_proba + 
                  weights[1] * rf_proba + 
                  weights[2] * gb_proba)

# Optimize threshold for low false positives
print("\nOptimizing decision threshold for low false positives...")

thresholds = np.arange(0.3, 0.9, 0.05)
best_threshold = 0.5
best_f1 = 0
results = []

for threshold in thresholds:
    y_pred = (ensemble_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # We want high precision (low false positives) while maintaining reasonable recall
    if precision >= 0.85 and f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
    
    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

print(f"\nOptimal threshold: {best_threshold:.2f}")

# Final predictions with optimal threshold
y_pred_final = (ensemble_proba >= best_threshold).astype(int)

# ==================== MODEL EVALUATION ====================
print("\n" + "=" * 70)
print("MODEL EVALUATION - ENSEMBLE WITH OPTIMIZED THRESHOLD")
print("=" * 70)

# Calculate metrics
accuracy = (y_pred_final == y_test).sum() / len(y_test)
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)
roc_auc = roc_auc_score(y_test, ensemble_proba)
avg_precision = average_precision_score(y_test, ensemble_proba)

print(f"\n{'Metric':<25} {'Score':<10}")
print("-" * 35)
print(f"{'Accuracy':<25} {accuracy:.4f}")
print(f"{'Precision (Low FP!)':<25} {precision:.4f}")
print(f"{'Recall':<25} {recall:.4f}")
print(f"{'F1-Score':<25} {f1:.4f}")
print(f"{'ROC-AUC':<25} {roc_auc:.4f}")
print(f"{'Average Precision':<25} {avg_precision:.4f}")

print(f"\n\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_final, 
                          target_names=['Legitimate', 'Fraud'],
                          digits=4))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)
print(f"\nTrue Negatives (Correct Legitimate): {cm[0,0]:,}")
print(f"False Positives (Legitimate flagged as Fraud): {cm[0,1]:,}")
print(f"False Negatives (Fraud missed): {cm[1,0]:,}")
print(f"True Positives (Correct Fraud detection): {cm[1,1]:,}")

false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1])
false_negative_rate = cm[1,0] / (cm[1,0] + cm[1,1])

print(f"\n✓ False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
print(f"✓ False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")

# Cost analysis
cost_fp = 10  # Cost of investigating false positive
cost_fn = 500  # Average fraud amount
total_cost = (cm[0,1] * cost_fp) + (cm[1,0] * cost_fn)
print(f"\nEstimated Cost Analysis:")
print(f"False Positive Cost: ${cm[0,1] * cost_fp:,.2f}")
print(f"False Negative Cost: ${cm[1,0] * cost_fn:,.2f}")
print(f"Total Operational Cost: ${total_cost:,.2f}")

# ==================== FEATURE IMPORTANCE ====================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# XGBoost feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features (XGBoost):")
print(feature_importance.head(15).to_string(index=False))

# ==================== VISUALIZATIONS ====================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 14))

# 1. Feature Importance
ax1 = plt.subplot(3, 3, 1)
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score')
plt.title('Top 15 Feature Importance', fontweight='bold', fontsize=12)
plt.gca().invert_yaxis()

# 2. Confusion Matrix
ax2 = plt.subplot(3, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix', fontweight='bold', fontsize=12)

# 3. ROC Curve
ax3 = plt.subplot(3, 3, 3)
fpr, tpr, _ = roc_curve(y_test, ensemble_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'Ensemble (AUC={roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Precision-Recall Curve
ax4 = plt.subplot(3, 3, 4)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, ensemble_proba)
plt.plot(recall_curve, precision_curve, linewidth=2, color='green',
         label=f'AP={avg_precision:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Threshold Analysis
ax5 = plt.subplot(3, 3, 5)
threshold_df = pd.DataFrame(results)
plt.plot(threshold_df['threshold'], threshold_df['precision'], 
         label='Precision', linewidth=2, marker='o')
plt.plot(threshold_df['threshold'], threshold_df['recall'], 
         label='Recall', linewidth=2, marker='s')
plt.plot(threshold_df['threshold'], threshold_df['f1'], 
         label='F1-Score', linewidth=2, marker='^')
plt.axvline(x=best_threshold, color='red', linestyle='--', 
            label=f'Optimal ({best_threshold:.2f})')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Threshold Optimization', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Prediction Distribution
ax6 = plt.subplot(3, 3, 6)
plt.hist(ensemble_proba[y_test == 0], bins=50, alpha=0.6, 
         label='Legitimate', color='blue', density=True)
plt.hist(ensemble_proba[y_test == 1], bins=50, alpha=0.6, 
         label='Fraud', color='red', density=True)
plt.axvline(x=best_threshold, color='green', linestyle='--', 
            linewidth=2, label=f'Threshold={best_threshold:.2f}')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Prediction Distribution', fontweight='bold', fontsize=12)
plt.legend()

# 7. Transaction Amount Distribution
ax7 = plt.subplot(3, 3, 7)
df[df['is_fraud'] == 0]['amount'].hist(bins=50, alpha=0.6, 
                                        label='Legitimate', color='blue')
df[df['is_fraud'] == 1]['amount'].hist(bins=50, alpha=0.6, 
                                        label='Fraud', color='red')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Frequency')
plt.title('Amount Distribution by Class', fontweight='bold', fontsize=12)
plt.legend()
plt.xlim(0, 5000)

# 8. Model Comparison
ax8 = plt.subplot(3, 3, 8)
models = ['XGBoost', 'Random Forest', 'Gradient Boosting', 'Ensemble']
xgb_pred = xgb_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test_scaled)
gb_pred = gb_model.predict(X_test_scaled)

precisions = [
    precision_score(y_test, xgb_pred),
    precision_score(y_test, rf_pred),
    precision_score(y_test, gb_pred),
    precision
]
recalls = [
    recall_score(y_test, xgb_pred),
    recall_score(y_test, rf_pred),
    recall_score(y_test, gb_pred),
    recall
]

x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, precisions, width, label='Precision', color='skyblue')
plt.bar(x + width/2, recalls, width, label='Recall', color='lightcoral')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison', fontweight='bold', fontsize=12)
plt.xticks(x, models, rotation=15, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 9. Time-based Analysis
ax9 = plt.subplot(3, 3, 9)
hour_fraud = df.groupby('hour')['is_fraud'].mean() * 100
plt.plot(hour_fraud.index, hour_fraud.values, marker='o', 
         linewidth=2, markersize=6, color='crimson')
plt.xlabel('Hour of Day')
plt.ylabel('Fraud Rate (%)')
plt.title('Fraud Rate by Hour', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig('fraud_detection_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'fraud_detection_analysis.png'")

# ==================== SAVE MODEL ====================
print("\n" + "=" * 70)
print("SAVING MODEL AND ARTIFACTS")
print("=" * 70)

import pickle

model_artifacts = {
    'xgb_model': xgb_model,
    'rf_model': rf_model,
    'gb_model': gb_model,
    'scaler': scaler,
    'feature_columns': feature_cols,
    'weights': weights,
    'threshold': best_threshold,
    'category_mapping': category_mapping
}

with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("\n✓ Model ensemble saved as 'fraud_detection_model.pkl'")
feature_importance.to_csv('fraud_feature_importance.csv', index=False)
print("✓ Feature importance saved as 'fraud_feature_importance.csv'")

# ==================== REAL-TIME PREDICTION FUNCTION ====================
print("\n" + "=" * 70)
print("REAL-TIME PREDICTION SYSTEM")
print("=" * 70)

def predict_fraud_realtime(transaction_dict, model_artifacts):
    """
    Real-time fraud prediction for a single transaction
    
    Parameters:
    -----------
    transaction_dict : dict
        Dictionary containing transaction features
    model_artifacts : dict
        Loaded model artifacts
    
    Returns:
    --------
    dict : Prediction results with fraud probability and decision
    """
    
    # Extract models and components
    xgb_model = model_artifacts['xgb_model']
    rf_model = model_artifacts['rf_model']
    gb_model = model_artifacts['gb_model']
    scaler = model_artifacts['scaler']
    feature_cols = model_artifacts['feature_columns']
    weights = model_artifacts['weights']
    threshold = model_artifacts['threshold']
    
    # Create DataFrame
    transaction_df = pd.DataFrame([transaction_dict])
    
    # Extract features in correct order
    X_new = transaction_df[feature_cols].values
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Get predictions from all models
    xgb_proba = xgb_model.predict_
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
import joblib
import warnings

warnings.filterwarnings('ignore')

INPUT_FILE = r"c:\Users\User\Documents\ttrsqr\featured_dataset.csv"
MODEL_FILE = r"c:\Users\User\Documents\ttrsqr\catboost_model.cbm"

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Ensure datetime for sorting
    df['transdatetime'] = pd.to_datetime(df['transdatetime'])
    df = df.sort_values('transdatetime').reset_index(drop=True)
    
    return df

def train_model(df):
    print("Preparing data for training...")
    
    # Define features and target
    target_col = 'target'
    
    # Drop non-feature columns
    drop_cols = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'direction', 'target']
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # Identify categorical features
    cat_features = [col for col in X.columns if X[col].dtype == 'object']
    print(f"Categorical features: {cat_features}")
    
    # Fill NaNs
    for col in cat_features:
        X[col] = X[col].fillna('MISSING')
    
    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold = 1
    all_metrics = []
    
    model = None
    
    print(f"\n{'='*80}")
    print("STARTING TIME SERIES CROSS-VALIDATION (5 FOLDS)")
    print(f"{'='*80}\n")
    
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Check if we have both classes in training data
        if len(y_train.unique()) < 2:
            print(f"Fold {fold} - Skipping: Training set has only 1 class.")
            fold += 1
            continue
        
        print(f"\n--- Fold {fold} ---")
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        print(f"Val fraud ratio: {y_val.sum() / len(y_val) * 100:.2f}%")
            
        # Initialize CatBoost
        clf = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            eval_metric='AUC',
            auto_class_weights='Balanced',
            cat_features=cat_features,
            verbose=200,
            early_stopping_rounds=50,
            random_seed=42
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # Predictions
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)
        
        # Calculate comprehensive metrics
        from sklearn.metrics import (
            precision_score, recall_score, fbeta_score, 
            confusion_matrix, roc_auc_score, average_precision_score
        )
        
        metrics = {
            'fold': fold,
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'pr_auc': average_precision_score(y_val, y_pred_proba),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1_score': fbeta_score(y_val, y_pred, beta=1, zero_division=0),
            'f2_score': fbeta_score(y_val, y_pred, beta=2, zero_division=0),  # Emphasizes recall
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }
        
        all_metrics.append(metrics)
        
        # Print fold metrics
        print(f"\nFold {fold} Results:")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  F2-Score:  {metrics['f2_score']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {cm[0][0]:5d} | FP: {cm[0][1]:5d}")
        print(f"    FN: {cm[1][0]:5d} | TP: {cm[1][1]:5d}")
        
        model = clf
        fold += 1
    
    # Calculate average metrics
    print(f"\n{'='*80}")
    print("AVERAGE METRICS ACROSS ALL FOLDS")
    print(f"{'='*80}\n")
    
    avg_metrics = {
        'roc_auc': np.mean([m['roc_auc'] for m in all_metrics]),
        'pr_auc': np.mean([m['pr_auc'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'f2_score': np.mean([m['f2_score'] for m in all_metrics]),
    }
    
    for metric, value in avg_metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")
    
    # Save detailed metrics to JSON
    import json
    metrics_file = r"c:\Users\User\Documents\ttrsqr\model_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'fold_metrics': all_metrics,
            'average_metrics': avg_metrics
        }, f, indent=2)
    print(f"\nDetailed metrics saved to: {metrics_file}")
    
    # Final training on ALL data
    print("\n" + "="*80)
    print("RETRAINING ON FULL DATASET FOR PRODUCTION MODEL")
    print("="*80 + "\n")
    
    final_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        auto_class_weights='Balanced',
        cat_features=cat_features,
        verbose=200,
        random_seed=42
    )
    final_model.fit(X, y)
    
    return final_model, X.columns.tolist(), all_metrics

def main():
    df = load_data(INPUT_FILE)
    
    model, feature_names, all_metrics = train_model(df)
    
    print(f"\nSaving model to {MODEL_FILE}...")
    model.save_model(MODEL_FILE)
    
    # Save feature importance
    importance = model.get_feature_importance()
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feat_imp.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Metrics saved to: c:\\Users\\User\\Documents\\ttrsqr\\model_metrics.json")
    print("\nDone!")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score, precision_score, recall_score, fbeta_score, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Получаем директорию текущего скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_DIR, "featured_dataset.csv")
MODEL_FILE = os.path.join(SCRIPT_DIR, "lgbm_model.pkl") # Changed extension for LGBM

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Ensure datetime for sorting (though we use StratifiedKFold now, sorting is good practice)
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
    
    # Clean feature names for LightGBM (remove special chars)
    import re
    X.columns = [re.sub(r'[^\w]', '_', col) for col in X.columns]
    print("Feature names cleaned.")
    
    # Identify categorical features
    cat_features = [col for col in X.columns if X[col].dtype == 'object']
    print(f"Categorical features: {cat_features}")
    
    # LightGBM handles categorical features internally if they are 'category' type
    for col in cat_features:
        X[col] = X[col].astype('category')
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold = 1
    all_metrics = []
    
    print(f"\n{'='*80}")
    print("STARTING STRATIFIED CROSS-VALIDATION (5 FOLDS)")
    print(f"{'='*80}\n")
    
    # Placeholder for best params found in first fold
    best_params = {}
    
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        print(f"\n--- Fold {fold} ---")
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        print(f"Val fraud ratio: {y_val.sum() / len(y_val) * 100:.2f}%")
            
        # Hyperparameter Tuning with RandomizedSearchCV
        if fold == 1:
            print("  Tuning hyperparameters (RandomizedSearchCV)...")
            param_dist = {
                'n_estimators': [500, 1000, 2000],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'num_leaves': [20, 31, 50, 100],
                'max_depth': [-1, 10, 20],
                'reg_alpha': [0, 0.1, 1, 10],
                'reg_lambda': [0, 0.1, 1, 10],
                'min_child_samples': [10, 20, 50],
                'subsample': [0.7, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.9, 1.0]
            }
            
            lgbm = LGBMClassifier(
                objective='binary',
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
            
            random_search = RandomizedSearchCV(
                lgbm, 
                param_distributions=param_dist, 
                n_iter=20, # 20 iterations
                scoring='f1', 
                cv=3, 
                verbose=1, 
                random_state=42,
                n_jobs=-1
            )
            
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            print(f"  Best Params: {best_params}")
        
        # Train with best params
        clf = LGBMClassifier(
            **best_params,
            objective='binary',
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[] 
        )
        
        # Predictions
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)
        
        # Metrics
        metrics = {
            'fold': fold,
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'pr_auc': average_precision_score(y_val, y_pred_proba),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, zero_division=0),
            'f2_score': fbeta_score(y_val, y_pred, beta=2, zero_division=0),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }
        
        all_metrics.append(metrics)
        
        print(f"\nFold {fold} Results:")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix: TN:{cm[0][0]} FP:{cm[0][1]} FN:{cm[1][0]} TP:{cm[1][1]}")
        
        fold += 1
    
    # Average Metrics
    print(f"\n{'='*80}")
    print("AVERAGE METRICS")
    print(f"{'='*80}\n")
    
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0] if k != 'fold' and k != 'confusion_matrix'}
    for k, v in avg_metrics.items():
        print(f"{k.upper():15s}: {v:.4f}")
        
    # Feature Importance (Permutation)
    print("\nCalculating Permutation Importance (on last validation fold)...")
    perm_importance = permutation_importance(clf, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    
    sorted_idx = perm_importance.importances_mean.argsort()
    
    print("\nTop 10 Features by Permutation Importance:")
    top_features = []
    for i in sorted_idx[-10:]:
        print(f"{X.columns[i]}: {perm_importance.importances_mean[i]:.4f}")
        top_features.append(X.columns[i])
        
    # Save metrics
    metrics_file = os.path.join(SCRIPT_DIR, "model_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            'fold_metrics': all_metrics,
            'average_metrics': avg_metrics,
            'top_features': top_features
        }, f, indent=2)
        
    # Retrain on full data
    print("\nRetraining on full dataset...")
    final_model = LGBMClassifier(
        **best_params,
        objective='binary',
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    final_model.fit(X, y)
    
    return final_model

def main():
    df = load_data(INPUT_FILE)
    model = train_model(df)
    
    print(f"\nSaving model to {MODEL_FILE}...")
    joblib.dump(model, MODEL_FILE)
    print("Done!")

if __name__ == "__main__":
    main()

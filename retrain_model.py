import os
import shutil
from datetime import datetime
import json
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# Получаем директорию текущего скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = SCRIPT_DIR
MODELS_DIR = os.path.join(DATA_DIR, "models")
FEATURED_DATASET = os.path.join(DATA_DIR, "featured_dataset.csv")
PRODUCTION_MODEL = os.path.join(DATA_DIR, "catboost_model.cbm")

# Model versioning
os.makedirs(MODELS_DIR, exist_ok=True)

class ModelRetrainer:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.version_log_path = os.path.join(self.models_dir, "version_log.json")
        self.load_version_log()
    
    def load_version_log(self):
        """Load model version history"""
        if os.path.exists(self.version_log_path):
            with open(self.version_log_path, 'r') as f:
                self.version_log = json.load(f)
        else:
            self.version_log = {"versions": [], "current_version": None}
    
    def save_version_log(self):
        """Save model version history"""
        with open(self.version_log_path, 'w') as f:
            json.dump(self.version_log, f, indent=2)
    
    def get_next_version(self):
        """Generate next version number"""
        if not self.version_log["versions"]:
            return "v1.0.0"
        
        last_version = self.version_log["versions"][-1]["version"]
        # Simple increment: v1.0.0 -> v1.1.0
        parts = last_version.replace('v', '').split('.')
        parts[1] = str(int(parts[1]) + 1)
        return f"v{'.'.join(parts)}"
    
    def train_new_model(self, df):
        """Train a new model version"""
        print(f"\n{'='*80}")
        print("TRAINING NEW MODEL VERSION")
        print(f"{'='*80}\n")
        
        # Prepare data
        drop_cols = ['cst_dim_id', 'transdate', 'transdatetime', 'docno', 'direction', 'target']
        X = df.drop(columns=drop_cols)
        y = df['target']
        
        cat_features = [col for col in X.columns if X[col].dtype == 'object']
        for col in cat_features:
            X[col] = X[col].fillna('MISSING')
        
        # Quick validation with TimeSeriesSplit (1 fold for speed)
        tscv = TimeSeriesSplit(n_splits=2)
        train_idx, val_idx = list(tscv.split(X))[-1]
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = CatBoostClassifier(
            iterations=500,  # Reduced for faster retraining
            learning_rate=0.05,
            depth=6,
            eval_metric='AUC',
            auto_class_weights='Balanced',
            cat_features=cat_features,
            verbose=100,
            random_seed=42
        )
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        metrics = {
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0)
        }
        
        print(f"\nValidation Metrics:")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        
        # Retrain on full dataset
        print("\nRetraining on full dataset...")
        final_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            eval_metric='AUC',
            auto_class_weights='Balanced',
            cat_features=cat_features,
            verbose=False,
            random_seed=42
        )
        final_model.fit(X, y)
        
        return final_model, metrics
    
    def save_model_version(self, model, metrics, notes=""):
        """Save new model version with metadata"""
        version = self.get_next_version()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save model file
        model_filename = f"catboost_model_{version}.cbm"
        model_path = os.path.join(self.models_dir, model_filename)
        model.save_model(model_path)
        
        # Add to version log
        version_info = {
            "version": version,
            "timestamp": timestamp,
            "model_path": model_path,
            "metrics": metrics,
            "notes": notes
        }
        
        self.version_log["versions"].append(version_info)
        self.version_log["current_version"] = version
        self.save_version_log()
        
        print(f"\n[OK] Model {version} saved successfully!")
        print(f"   Path: {model_path}")
        print(f"   Timestamp: {timestamp}")
        
        return version, model_path
    
    def promote_to_production(self, version=None):
        """Promote a model version to production (replace catboost_model.cbm)"""
        if version is None:
            version = self.version_log["current_version"]
        
        # Find version in log
        version_info = None
        for v in self.version_log["versions"]:
            if v["version"] == version:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version {version} not found!")
        
        # Backup current production model
        if os.path.exists(PRODUCTION_MODEL):
            backup_path = os.path.join(self.models_dir, f"production_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.cbm")
            shutil.copy(PRODUCTION_MODEL, backup_path)
            print(f"[BACKUP] Backed up current production model to: {backup_path}")
        
        # Copy new version to production
        shutil.copy(version_info["model_path"], PRODUCTION_MODEL)
        print(f"[DEPLOYED] Promoted {version} to production!")
        print(f"   ROC-AUC: {version_info['metrics']['roc_auc']:.4f}")
        
        return True
    
    def retrain_and_deploy(self, auto_deploy=True):
        """Full retraining pipeline"""
        print(f"\n{'='*80}")
        print("AUTOMATED MODEL RETRAINING PIPELINE")
        print(f"{'='*80}\n")
        
        # Load data
        print(f"Loading data from {FEATURED_DATASET}...")
        df = pd.read_csv(FEATURED_DATASET)
        print(f"Dataset size: {len(df)} rows")
        
        # Train new model
        model, metrics = self.train_new_model(df)
        
        # Save version
        notes = f"Automated retraining on {len(df)} samples"
        version, model_path = self.save_model_version(model, metrics, notes)
        
        # Auto-deploy to production
        if auto_deploy:
            # Check if new model is better than current
            current_version = self.version_log.get("current_version")
            if len(self.version_log["versions"]) > 1:
                prev_metrics = self.version_log["versions"][-2]["metrics"]
                if metrics['roc_auc'] >= prev_metrics['roc_auc']:
                    print(f"\n[OK] New model is better or equal (ROC-AUC: {metrics['roc_auc']:.4f} >= {prev_metrics['roc_auc']:.4f})")
                    self.promote_to_production(version)
                else:
                    print(f"\n[WARNING] New model is worse (ROC-AUC: {metrics['roc_auc']:.4f} < {prev_metrics['roc_auc']:.4f})")
                    print("   Skipping deployment. Manual review required.")
            else:
                # First model, deploy automatically
                self.promote_to_production(version)
        
        print(f"\n{'='*80}")
        print("RETRAINING COMPLETED!")
        print(f"{'='*80}\n")
        
        self.print_version_history()
    
    def print_version_history(self):
        """Print all model versions"""
        print("\n" + "="*80)
        print("MODEL VERSION HISTORY")
        print("="*80 + "\n")
        
        if not self.version_log["versions"]:
            print("No versions found.")
            return
        
        for v in self.version_log["versions"]:
            is_current = "[CURRENT]" if v["version"] == self.version_log["current_version"] else ""
            print(f"{v['version']} {is_current}")
            print(f"  Timestamp: {v['timestamp']}")
            print(f"  ROC-AUC: {v['metrics']['roc_auc']:.4f}, Precision: {v['metrics']['precision']:.4f}, Recall: {v['metrics']['recall']:.4f}")
            print(f"  Notes: {v['notes']}")
            print()

if __name__ == "__main__":
    retrainer = ModelRetrainer()
    
    # Run retraining pipeline
    retrainer.retrain_and_deploy(auto_deploy=True)

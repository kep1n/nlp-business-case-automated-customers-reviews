"""
Sentiment Classification Fine-tuning Pipeline

Fine-tune transformer models (BERT/DistilBERT) for sentiment analysis
using PEFT/LoRA for efficient training.
"""

import os
import sqlite3
import warnings
from typing import Optional, Tuple, List, Dict, Literal
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
import mlflow
import mlflow.transformers
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()


@dataclass
class ModelConfig:
    """Model configuration"""
    DISTILBERT = "distilbert/distilbert-base-uncased"
    BERT = "google-bert/bert-base-uncased"
    
    LORA_TARGETS = {
        "distilbert": ["q_lin", "v_lin"],
        "bert": ["query", "value"]
    }


class DatabaseLoader:
    """Handle SQLite database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def load_split(self, split: str, text_col: str = 'review_text_clean', 
                   label_col: str = 'voted_up') -> Tuple[List[str], List[int]]:
        """Load data from specific split (train/validation/test)"""
        query = f"SELECT {text_col}, {label_col} FROM {split}"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        
        print(f"✓ Loaded {len(df)} samples from {split}")
        print(f"  Positive: {df[label_col].sum()}, Negative: {(~df[label_col].astype(bool)).sum()}")
        
        return df[text_col].tolist(), df[label_col].tolist()


class SentimentClassifier:
    """Fine-tune transformer models with LoRA for sentiment classification"""
    
    def __init__(self, model_type: Literal["bert", "distilbert"] = "distilbert",
                 use_lora: bool = True, num_labels: int = 2):
        self.model_type = model_type
        self.model_name = getattr(ModelConfig, model_type.upper())
        self.use_lora = use_lora
        self.num_labels = num_labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = None
        self.model = None
        
        print(f"Initializing {model_type.upper()} classifier")
        print(f"Device: {self.device}")
        print(f"LoRA enabled: {use_lora}")
    
    def _prepare_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """Convert lists to HuggingFace Dataset"""
        return Dataset.from_dict({'text': texts, 'label': labels})
    
    def _tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize batch of texts"""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _setup_lora(self) -> LoraConfig:
        """Configure LoRA parameters"""
        target_modules = ModelConfig.LORA_TARGETS[self.model_type]
        
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
            inference_mode=False
        )
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int],
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 3e-4,
              output_dir: str = './results') -> Trainer:
        """Fine-tune model"""
        
        print(f"\n{'='*70}")
        print(f"TRAINING {self.model_name}")
        print(f"{'='*70}")
        print(f"Training: {len(train_texts)} | Validation: {len(val_texts)}")
        print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
        
        # Load model and tokenizer
        print("\nLoading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Apply LoRA
        if self.use_lora:
            print("Applying LoRA...")
            lora_config = self._setup_lora()
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.model = self.model.to(self.device)
        
        # Prepare datasets
        print("Preparing datasets...")
        train_dataset = self._prepare_dataset(train_texts, train_labels)
        val_dataset = self._prepare_dataset(val_texts, val_labels)
        
        train_dataset = train_dataset.map(self._tokenize_function, batched=True)
        val_dataset = val_dataset.map(self._tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
            fp16=torch.cuda.is_available()
        )
        
        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        print("\nTraining started...")
        trainer.train()
        print("\n✓ Training complete")
        
        # Save
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✓ Model saved to {output_dir}")
        
        return trainer
    
    def load_model(self, model_path: str, base_model: Optional[str] = None) -> None:
        """Load fine-tuned model"""
        print(f"\nLoading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.use_lora:
            base_model = base_model or self.model_name
            base = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=self.num_labels
            )
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded")
    
    def predict(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with probabilities"""
        self.model.eval()
        predictions, probabilities = [], []
        
        print(f"Predicting on {len(texts)} samples...")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                
                predictions.extend(preds)
                probabilities.extend(probs)
        
        print("✓ Predictions complete")
        return np.array(predictions), np.array(probabilities)


class ModelEvaluator:
    """Evaluate sentiment classification models"""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                 class_names: List[str] = None,
                 save_path: Optional[str] = None) -> Dict:
        """Comprehensive model evaluation"""
        
        class_names = class_names or ['Negative', 'Positive']
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        print(f"\nMacro Averages:")
        print(f"  Precision: {precision_macro*100:.2f}%")
        print(f"  Recall:    {recall_macro*100:.2f}%")
        print(f"  F1-Score:  {f1_macro*100:.2f}%")
        
        print("\nPer-Class Metrics:")
        for i, name in enumerate(class_names):
            print(f"\n{name} (n={support[i]}):")
            print(f"  Precision: {precision[i]*100:.2f}%")
            print(f"  Recall:    {recall[i]*100:.2f}%")
            print(f"  F1-Score:  {f1[i]*100:.2f}%")
        
        print("\n" + classification_report(y_true, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        if save_path:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Confusion matrix saved to {save_path}")
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm
        }


class SentimentFineTuningPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, db_path: str, model_type: Literal["bert", "distilbert"] = "distilbert",
                 use_lora: bool = True, experiment_name: str = "sentiment-fine-tune"):
        
        self.db_loader = DatabaseLoader(db_path)
        self.classifier = SentimentClassifier(model_type=model_type, use_lora=use_lora)
        self.evaluator = ModelEvaluator()
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        mlflow.autolog()
    
    def run(self, epochs: int = 3, batch_size: int = 16, 
            learning_rate: float = 3e-4, output_dir: str = './model_output') -> Dict:
        """Execute complete fine-tuning pipeline"""
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", self.classifier.model_type)
            mlflow.log_param("use_lora", self.classifier.use_lora)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            
            print("\n" + "="*70)
            print("SENTIMENT FINE-TUNING PIPELINE")
            print("="*70)
            
            # Load data
            print("\nSTEP 1: Loading data...")
            train_texts, train_labels = self.db_loader.load_split('train')
            val_texts, val_labels = self.db_loader.load_split('validation')
            test_texts, test_labels = self.db_loader.load_split('test')
            
            mlflow.log_metric("train_samples", len(train_texts))
            mlflow.log_metric("val_samples", len(val_texts))
            mlflow.log_metric("test_samples", len(test_texts))
            
            # Train
            print("\nSTEP 2: Training model...")
            self.classifier.train(
                train_texts, train_labels,
                val_texts, val_labels,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_dir=output_dir
            )
            
            # Evaluate
            print("\nSTEP 3: Evaluating model...")
            predictions, probabilities = self.classifier.predict(test_texts)
            
            cm_path = Path(output_dir) / 'confusion_matrix.png'
            results = self.evaluator.evaluate(
                np.array(test_labels),
                predictions,
                save_path=str(cm_path)
            )
            
            # Log metrics
            mlflow.log_metric("test_accuracy", results['accuracy'])
            mlflow.log_metric("test_f1_macro", results['f1_macro'])
            mlflow.log_artifact(str(cm_path))
            
            # Test loading
            print("\nSTEP 4: Testing model loading...")
            classifier_loaded = SentimentClassifier(
                model_type=self.classifier.model_type,
                use_lora=self.classifier.use_lora
            )
            classifier_loaded.load_model(output_dir)
            loaded_preds, _ = classifier_loaded.predict(test_texts[:100])
            
            match = np.array_equal(predictions[:100], loaded_preds)
            print(f"✓ Model reload verification: {'PASSED' if match else 'FAILED'}")
            
            print("\n" + "="*70)
            print("PIPELINE COMPLETE")
            print("="*70)
            print(f"Model saved to: {output_dir}")
            print(f"Test Accuracy: {results['accuracy']*100:.2f}%")
            print(f"Test F1 (Macro): {results['f1_macro']*100:.2f}%")
            
            return results


def main():
    """Main entry point"""
    
    # Configuration
    DB_PATH = '/content/reviews_processed.db'
    MODEL_TYPE = "distilbert"  # Options: "bert", "distilbert"
    USE_LORA = True
    EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4 if USE_LORA else 2e-5
    OUTPUT_DIR = f'./model_{MODEL_TYPE}_lora' if USE_LORA else f'./model_{MODEL_TYPE}_full'
    
    # Setup MLflow credentials (optional)
    # os.environ['MLFLOW_TRACKING_URI'] = 'your_uri'
    # os.environ['MLFLOW_TRACKING_USERNAME'] = 'username'
    # os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
    
    # Run pipeline
    pipeline = SentimentFineTuningPipeline(
        db_path=DB_PATH,
        model_type=MODEL_TYPE,
        use_lora=USE_LORA
    )
    
    results = pipeline.run(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
# src/models/evaluate.py

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator


class ModelEvaluator:
    """Evaluate trained model and generate reports"""
    
    def __init__(self, model_path="./models/bert-rakuten-final"):
        self.model_path = model_path
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load label encoder
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            self.le = pickle.load(f)
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get id2label from model config
        self.id2label = self.model.config.id2label
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
    
    def get_predictions(self, dataloader):
        """Get predictions for entire dataset"""
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        labels = np.concatenate(all_labels)
        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)
        
        return labels, predictions, probabilities
    
    def generate_classification_report(self, labels, predictions, save_path=None):
        """Generate and optionally save classification report"""
        # Get class names
        target_names = [self.id2label[i] for i in range(len(self.id2label))]
        
        # Generate report
        report_dict = classification_report(
            labels, predictions, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        report_str = classification_report(
            labels, predictions,
            target_names=target_names,
            zero_division=0
        )
        
        print("Classification Report:")
        print(report_str)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
            print(f"Classification report saved to {save_path}")
        
        return report_dict
    
    def generate_confusion_matrix(self, labels, predictions, save_path=None):
        """Generate and optionally save confusion matrix visualization"""
        # Get class names
        ticks = [self.id2label[i] for i in range(len(self.id2label))]
        
        # Compute confusion matrix (normalized)
        cm = confusion_matrix(labels, predictions, normalize='true')
        cm_display = np.round(cm, 2)
        
        # Create figure
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            cm_display, 
            annot=True, 
            fmt='.2f', 
            cmap='viridis',
            xticklabels=ticks, 
            yticklabels=ticks,
            cbar_kws={'label': 'Normalized frequency'}
        )
        plt.title("Confusion Matrix (Normalized)", fontsize=16)
        plt.xlabel('Predicted Category', fontsize=12)
        plt.ylabel('True Category', fontsize=12)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        return cm
    
    def evaluate_dataset(self, dataset, batch_size=16, output_dir="./results/evaluation"):
        """Complete evaluation pipeline"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*50}\nEVALUATING MODEL\n{'='*50}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator
        )
        
        # Get predictions
        print("Generating predictions...")
        labels, predictions, probabilities = self.get_predictions(dataloader)
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report_path = f"{output_dir}/classification_report_{timestamp}.json"
        report = self.generate_classification_report(labels, predictions, report_path)
        
        # Generate confusion matrix
        cm_path = f"{output_dir}/confusion_matrix_{timestamp}.png"
        cm = self.generate_confusion_matrix(labels, predictions, cm_path)
        
        # Save predictions
        predictions_data = {
            "labels": labels.tolist(),
            "predictions": predictions.tolist(),
            "accuracy": float(accuracy),
            "timestamp": timestamp,
            "model_path": self.model_path
        }
        
        pred_path = f"{output_dir}/predictions_{timestamp}.json"
        with open(pred_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f"Predictions saved to {pred_path}")
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": predictions_data
        }


# CLI usage
if __name__ == "__main__":
    import argparse
    from datasets import load_from_disk
    
    parser = argparse.ArgumentParser(description="Evaluate BERT model")
    parser.add_argument("--model_path", type=str, default="./models/bert-rakuten-final",
                       help="Path to trained model")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to saved tokenized dataset (optional)")
    parser.add_argument("--output_dir", type=str, default="./results/evaluation",
                       help="Output directory for evaluation results")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Load dataset (you'll need to save tokenized dataset or recreate it)
    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)
        # Check if it's a DatasetDict or single Dataset
        if hasattr(dataset, 'keys'):  # It's a DatasetDict
            test_dataset = dataset["test"]
        else:  # It's already a single Dataset
            test_dataset = dataset  # FIX: Use directly
    else:
        # Recreate dataset (import from train script or separate module)
        print("No dataset path provided. Load or recreate your test dataset.")
        import sys
        sys.exit(1)
    
    # Evaluate
    evaluator = ModelEvaluator(args.model_path)
    results = evaluator.evaluate_dataset(
        test_dataset,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*50}\nEVALUATION COMPLETE\n{'='*50}")

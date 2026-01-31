# src/evaluate/services/evaluate.py
"""
Model Evaluation Service

Evaluates trained BERT model on test data.
Generates classification reports, confusion matrices, and predictions.
Works with preprocessed parquet data from Data Service.
"""

# Set non-interactive backend BEFORE importing pyplot
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")  # Use non-interactive backend for background tasks
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from core.config import get_path
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator


class ModelEvaluator:
    """
    Evaluate trained BERT model and generate comprehensive reports.

    Features:
    - Classification report with per-class metrics
    - Confusion matrix visualization
    - Prediction export
    - Top-K accuracy
    - Confidence analysis
    """

    def __init__(self, model_path: Optional[str] = None, model_name: str = "bert-rakuten-final"):
        """
        Initialize model evaluator.

        Args:
            model_path: Full path to model directory. If None, uses model_name.
            model_name: Model directory name (used if model_path is None)
        """

        # Determine model path
        if model_path is None:
            models_dir = get_path("models.save_dir")
            model_path = models_dir / model_name

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"\n{'=' * 60}")
        print("INITIALIZING MODEL EVALUATOR")
        print(f"{'=' * 60}\n")

        # Device setup (same as training/inference)
        self.device = self._setup_device()

        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("✓ Model loaded")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        print("✓ Tokenizer loaded")

        # Load label encoder (created by Data Service)
        le_path = self.model_path.parent / "label_encoder.pkl"
        if not le_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at {le_path}. Make sure training pipeline saved it."
            )

        with open(le_path, "rb") as f:
            self.le = pickle.load(f)

        print(f"✓ Label encoder loaded ({len(self.le.classes_)} classes)")

        # Get label mappings from model config
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.num_labels = len(self.id2label)

        print("\n✓ Evaluator ready!")
        print(f"  Device: {self.device}")
        print(f"  Num labels: {self.num_labels}")

    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ Using Apple Metal (MPS) acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("✓ Using CUDA acceleration")
        else:
            device = torch.device("cpu")
            print("✓ Using CPU")

        return device

    def load_test_data(self, test_path: Optional[Path] = None) -> Dataset:
        """
        Load preprocessed test data from parquet file.

        Data is ALREADY:
        - Cleaned (HTML removed, encoding fixed)
        - Transformed (outliers handled)
        - Tokenized (BERT input_ids, attention_mask)
        - Encoded (labels are integers)

        Args:
            test_path: Path to test.parquet. If None, loads from config.

        Returns:
            tuple: (HuggingFace Dataset, pandas DataFrame)
        """
        if test_path is None:
            preprocessed_dir = get_path("data.preprocessed")
            test_path = preprocessed_dir / "test.parquet"

        if not test_path.exists():
            raise FileNotFoundError(
                f"Test data not found at {test_path}. Run Data Service preprocessing first."
            )

        # Load parquet
        test_df = pd.read_parquet(test_path)

        print(f"\n✓ Loaded test data: {len(test_df):,} samples")
        print(f"  Columns: {list(test_df.columns)}")

        # Convert to HuggingFace Dataset
        # Data is already tokenized by Data Service!
        test_cols = ["input_ids", "attention_mask", "labels"]

        missing_cols = [col for col in test_cols if col not in test_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Test data must be preprocessed with tokenization."
            )
        test_dataset = Dataset.from_pandas(test_df[test_cols])

        return test_dataset, test_df

    def get_predictions(
        self, dataloader: DataLoader, return_logits: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions for entire dataset.

        Args:
            dataloader: DataLoader with test data
            return_logits: If True, also return raw logits

        Returns:
            tuple: (labels, predictions, probabilities, [logits])
        """
        all_labels = []
        all_preds = []
        all_probs = []
        all_logits = [] if return_logits else None

        print("\nGenerating predictions...")

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                # Store results
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

                if return_logits:
                    all_logits.append(logits.cpu().numpy())

                # Progress
                if (i + 1) % 10 == 0:
                    print(f"  Processed {(i + 1) * dataloader.batch_size:,} samples...")

        # Concatenate results
        labels = np.concatenate(all_labels)
        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)

        print(f"✓ Generated {len(predictions):,} predictions")

        if return_logits:
            logits = np.concatenate(all_logits)
            return labels, predictions, probabilities, logits

        return labels, predictions, probabilities

    def generate_classification_report(
        self, labels: np.ndarray, predictions: np.ndarray, save_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate comprehensive classification report.

        Args:
            labels: Ground truth labels
            predictions: Predicted labels
            save_path: Path to save JSON report

        Returns:
            dict: Classification report
        """
        print(f"\n{'=' * 60}")
        print("CLASSIFICATION REPORT")
        print(f"{'=' * 60}\n")

        # Get class names (category IDs)
        target_names = [str(self.id2label[i]) for i in range(self.num_labels)]

        # Generate detailed report
        report_dict = classification_report(
            labels, predictions, target_names=target_names, output_dict=True, zero_division=0
        )

        report_str = classification_report(
            labels, predictions, target_names=target_names, zero_division=0, digits=4
        )

        print(report_str)

        # Calculate additional metrics
        overall_metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(labels, predictions, average="weighted", zero_division=0)
            ),
            "macro_precision": float(
                precision_score(labels, predictions, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(labels, predictions, average="macro", zero_division=0)
            ),
        }

        print("\nOverall Metrics:")
        for metric, value in overall_metrics.items():
            print(f"  {metric:20s}: {value:.4f}")

        # Combine into single report
        full_report = {"overall_metrics": overall_metrics, "per_class_metrics": report_dict}

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w") as f:
                json.dump(full_report, f, indent=2)
            print(f"\n✓ Classification report saved to {save_path}")

        return full_report

    def generate_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        save_path: Optional[Path] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate and visualize confusion matrix.

        Args:
            labels: Ground truth labels
            predictions: Predicted labels
            save_path: Path to save confusion matrix image
            normalize: If True, normalize by true label counts

        Returns:
            np.ndarray: Confusion matrix
        """
        print(f"\n{'=' * 60}")
        print("CONFUSION MATRIX")
        print(f"{'=' * 60}\n")

        # Get class names (category IDs)
        tick_labels = [str(self.id2label[i]) for i in range(self.num_labels)]

        # Compute confusion matrix
        cm = confusion_matrix(labels, predictions, normalize="true" if normalize else None)

        # Create visualization
        plt.figure(figsize=(max(20, self.num_labels), max(10, self.num_labels // 2)))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True if self.num_labels <= 30 else False,  # Skip annotations if too many classes
            fmt=".2f" if normalize else "d",
            cmap="viridis",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            cbar_kws={"label": "Normalized frequency" if normalize else "Count"},
            square=True,
        )

        plt.title(
            f"Confusion Matrix {'(Normalized)' if normalize else '(Counts)'}", fontsize=16, pad=20
        )
        plt.xlabel("Predicted Category", fontsize=14)
        plt.ylabel("True Category", fontsize=14)
        plt.tight_layout()

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Confusion matrix saved to {save_path}")

        plt.close()  # Don't show in non-interactive environments

        return cm

    def analyze_confidence(
        self, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
    ) -> Dict:
        """
        Analyze prediction confidence.

        Args:
            labels: Ground truth labels
            predictions: Predicted labels
            probabilities: Prediction probabilities (softmax outputs)

        Returns:
            dict: Confidence analysis
        """
        print(f"\n{'=' * 60}")
        print("CONFIDENCE ANALYSIS")
        print(f"{'=' * 60}\n")

        # Get max probabilities (confidence scores)
        confidences = probabilities.max(axis=1)

        # Separate correct and incorrect predictions
        correct_mask = predictions == labels
        incorrect_mask = ~correct_mask

        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[incorrect_mask]

        analysis = {
            "overall": {
                "mean_confidence": float(confidences.mean()),
                "median_confidence": float(np.median(confidences)),
                "std_confidence": float(confidences.std()),
                "min_confidence": float(confidences.min()),
                "max_confidence": float(confidences.max()),
            },
            "correct_predictions": {
                "count": int(correct_mask.sum()),
                "mean_confidence": float(correct_confidences.mean())
                if len(correct_confidences) > 0
                else 0.0,
                "median_confidence": float(np.median(correct_confidences))
                if len(correct_confidences) > 0
                else 0.0,
            },
            "incorrect_predictions": {
                "count": int(incorrect_mask.sum()),
                "mean_confidence": float(incorrect_confidences.mean())
                if len(incorrect_confidences) > 0
                else 0.0,
                "median_confidence": float(np.median(incorrect_confidences))
                if len(incorrect_confidences) > 0
                else 0.0,
            },
        }

        print("Overall Confidence:")
        print(f"  Mean:   {analysis['overall']['mean_confidence']:.4f}")
        print(f"  Median: {analysis['overall']['median_confidence']:.4f}")
        print(f"  Std:    {analysis['overall']['std_confidence']:.4f}")

        print("\nCorrect Predictions:")
        print(f"  Count:  {analysis['correct_predictions']['count']:,}")
        print(f"  Mean confidence: {analysis['correct_predictions']['mean_confidence']:.4f}")

        print("\nIncorrect Predictions:")
        print(f"  Count:  {analysis['incorrect_predictions']['count']:,}")
        print(f"  Mean confidence: {analysis['incorrect_predictions']['mean_confidence']:.4f}")

        return analysis

    def evaluate_dataset(
        self,
        dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        output_dir: Optional[str] = None,
        model_name: str = "bert-rakuten-final",
    ) -> Dict:
        """
        Complete evaluation pipeline.

        Args:
            dataset: HuggingFace Dataset to evaluate. If None, loads test.parquet.
            batch_size: Batch size for evaluation
            output_dir: Output directory for results. If None, uses config.

        Returns:
            dict: Complete evaluation results
        """
        print(f"\n{'=' * 60}")
        print("STARTING MODEL EVALUATION")
        print(f"{'=' * 60}\n")

        # Load test data if not provided
        if dataset is None:
            dataset, test_df = self.load_test_data()
        else:
            pass

        # Setup output directory
        if output_dir is None:
            output_dir = get_path("results.evaluation") / model_name

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Output directory: {output_dir}")
        print(f"Timestamp: {timestamp}")

        # Create dataloader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=default_data_collator, shuffle=False
        )

        print("\nEvaluation settings:")
        print(f"  Dataset size: {len(dataset):,}")
        print(f"  Batch size:   {batch_size}")
        print(f"  Num batches:  {len(dataloader)}")

        # Get predictions
        labels, predictions, probabilities = self.get_predictions(dataloader)

        # Generate classification report
        report_path = output_dir / f"classification_report_{timestamp}.json"
        report = self.generate_classification_report(labels, predictions, save_path=report_path)

        # Generate confusion matrix
        cm_path = output_dir / f"confusion_matrix_{timestamp}.png"
        cm = self.generate_confusion_matrix(labels, predictions, save_path=cm_path, normalize=True)

        # Analyze confidence
        confidence_analysis = self.analyze_confidence(labels, predictions, probabilities)

        # Save predictions with metadata
        predictions_data = {
            "metadata": {
                "timestamp": timestamp,
                "model_path": str(self.model_path),
                "dataset_size": len(dataset),
                "num_labels": self.num_labels,
            },
            "metrics": {
                "accuracy": float(accuracy_score(labels, predictions)),
                "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
                "weighted_f1": float(
                    f1_score(labels, predictions, average="weighted", zero_division=0)
                ),
            },
            "predictions": {
                "labels": labels.tolist(),
                "predictions": predictions.tolist(),
                "max_probabilities": probabilities.max(axis=1).tolist(),
            },
            "confidence_analysis": confidence_analysis,
        }

        pred_path = output_dir / f"predictions_{timestamp}.json"
        with open(pred_path, "w") as f:
            json.dump(predictions_data, f, indent=2)
        print(f"\n✓ Predictions saved to {pred_path}")

        # Create summary
        summary = {
            "timestamp": timestamp,
            "model_path": str(self.model_path),
            "dataset_size": len(dataset),
            "accuracy": float(accuracy_score(labels, predictions)),
            "classification_report": report,
            "confusion_matrix_shape": cm.shape,
            "confidence_analysis": confidence_analysis,
            "output_files": {
                "classification_report": str(report_path),
                "confusion_matrix": str(cm_path),
                "predictions": str(pred_path),
            },
        }

        # Save summary
        summary_path = output_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Evaluation summary saved to {summary_path}")

        print(f"\n{'=' * 60}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 60}\n")
        print(f"Results saved to: {output_dir}")

        return summary


################################################
### CLI EXECUTION
################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained BERT model on test data")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-rakuten-final",
        help="Model directory name (default: bert-rakuten-final)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Full path to model directory (overrides model_name)",
    )
    parser.add_argument(
        "--test_path", type=str, default=None, help="Path to test.parquet (default: from config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: from config)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation (default: 32)"
    )

    args = parser.parse_args()

    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path=args.model_path, model_name=args.model_name)

        # Load test data
        if args.test_path:
            test_dataset, _ = evaluator.load_test_data(Path(args.test_path))
        else:
            test_dataset, _ = evaluator.load_test_data()

        # Evaluate
        results = evaluator.evaluate_dataset(
            dataset=test_dataset, batch_size=args.batch_size, output_dir=args.output_dir
        )

        print("\n" + "=" * 60)
        print(" SUCCESS: Evaluation completed!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(" ERROR: Evaluation failed!")
        print("=" * 60)
        print(f"\nError details: {e}")
        raise

"""
Production inference service for product category prediction.

Uses shared preprocessing library for consistency with training. Handles device
management (CPU/CUDA/MPS) and batch prediction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from core.config import get_path, load_config
from transformers import AutoModelForSequenceClassification

from services.text_preparation_predict import TextPreparationPredict


class PredictionService:
    """
    Production prediction service for API endpoints.

    Features: - Automatic device detection (CPU/CUDA/MPS) - Single and batch
    prediction - Top-K probability predictions - Consistent preprocessing with
    training
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize prediction service.

        Args:
            model_path: Path to saved model directory. If None, loads from
            config.
        """
        # Load configuration
        self.config = load_config("params")

        # Determine model path
        if model_path is None:
            model_path = get_path("models.final")
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Device setup (keep from original - important for performance!)
        self.device = self._setup_device()

        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Initialize preprocessor WITH label encoder for decoding predictions
        label_encoder_path = self.model_path.parent / "label_encoder.pkl"
        if not label_encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at {label_encoder_path}. "
                f"Make sure training pipeline saved it."
            )

        self.text_preparation_predict = TextPreparationPredict.from_config(
            self.config["preprocessing"], label_encoder_path=str(label_encoder_path)
        )

        print("  Predictor ready!")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Num labels: {self.text_preparation_predict.get_num_labels()}")

    def _setup_device(self) -> torch.device:
        """
        Setup computation device with priority: MPS > CUDA > CPU

        Returns:
            torch.device: Selected device
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Metal (MPS) acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA acceleration (GPU: {torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("Using CPU (consider GPU for faster inference)")

        return device

    def predict(
        self, text: Union[str, List[str]], return_probabilities: bool = False, top_k: int = 5
    ) -> Union[Dict, List[Dict]]:
        """
        Predict category for single text or batch of texts.

        Args:
            text: Single text string or list of texts return_probabilities: If
            True, return top-K probabilities top_k: Number of top predictions to
            return (default: 5)

        Returns:
            Single dict or list of dicts with predictions

        """
        # Handle single text
        single_input = isinstance(text, str)
        if single_input:
            text = [text]

        # Preprocess texts using shared library (NO category_id - this is
        # inference!)
        preprocessed = []
        for txt in text:
            result = self.text_preparation_predict.preprocess_text(txt)
            preprocessed.append(result)

        # Extract input_ids and attention_masks
        input_ids = torch.tensor([p["input_ids"] for p in preprocessed])
        attention_mask = torch.tensor([p["attention_mask"] for p in preprocessed])

        # Move to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Get predictions
        pred_labels = torch.argmax(logits, dim=-1).cpu().numpy()
        confidences = probs.max(dim=-1).values.cpu().numpy()

        # Decode predictions using shared preprocessor
        pred_categories = [
            self.text_preparation_predict.decode_label(int(label)) for label in pred_labels
        ]

        # Build results
        results = []
        for i, (cat, conf, label, prep) in enumerate(
            zip(pred_categories, confidences, pred_labels, preprocessed)
        ):
            result = {
                "text": text[i],
                "cleaned_text": prep["cleaned_text"],
                "predicted_category": int(cat),
                "confidence": float(conf),
                "predicted_label": int(label),
            }

            # Add top-K probabilities if requested
            if return_probabilities:
                topk_probs, topk_indices = torch.topk(probs[i], k=min(top_k, len(probs[i])))

                result["top_predictions"] = [
                    {
                        "category": int(self.text_preparation_predict.decode_label(int(idx))),
                        "label": int(idx),
                        "probability": float(prob),
                    }
                    for prob, idx in zip(topk_probs.cpu().numpy(), topk_indices.cpu().numpy())
                ]

            results.append(result)

        # Return single dict if single input
        return results[0] if single_input else results

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_probabilities: bool = False,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Predict for large batches efficiently with mini-batching.

        Args:
            texts: List of texts to classify batch_size: Number of texts to
            process at once (adjust based on GPU memory) return_probabilities:
            If True, return top-K probabilities top_k: Number of top predictions
            to return

        Returns:
            List of prediction dictionaries

        """
        all_results = []

        print(f"Processing {len(texts)} texts in batches of {batch_size}...")

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = self.predict(
                batch, return_probabilities=return_probabilities, top_k=top_k
            )

            # Handle single result vs list
            if isinstance(batch_results, list):
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(texts)} texts...")

        print(f" Completed {len(all_results)} predictions")
        return all_results

    def predict_product(
        self,
        designation: str,
        description: str = "",
        return_probabilities: bool = False,
        top_k: int = 5,
    ) -> Dict:
        """
        Predict category for product with designation and description.

        Args:
            designation: Product title description: Product description
            (optional) return_probabilities: If True, return top-K probabilities
            top_k: Number of top predictions to return

        Returns:
            Prediction dictionary

        """
        # Use shared preprocessor to combine fields
        combined_text = self.text_preparation_predict._combine_fields(designation, description)

        result = self.predict(combined_text, return_probabilities=return_probabilities, top_k=top_k)

        # Add original fields to result
        result["designation"] = designation
        result["description"] = description

        return result


# ============================================================ CLI usage for
# testing
# ============================================================
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict product categories with BERT")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model directory (default: from config)",
    )
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument(
        "--file", type=str, help="JSON file with texts to classify (list of strings)"
    )
    parser.add_argument("--designation", type=str, help="Product designation (title)")
    parser.add_argument(
        "--description", type=str, default="", help="Product description (optional)"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Return top-K predictions (default: 5)"
    )
    parser.add_argument(
        "--probabilities", action="store_true", help="Return probability distributions"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for batch prediction (default: 32)"
    )

    args = parser.parse_args()

    # Initialize predictor
    print("Initializing predictor...")
    predictor = PredictionService(model_path=args.model_path)

    # Predict based on input type
    if args.text:
        # Single text prediction
        print(f"\nPredicting for text: '{args.text}'")
        result = predictor.predict(
            args.text, return_probabilities=args.probabilities, top_k=args.top_k
        )
        print("\nResult:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.designation:
        # Product prediction
        print("\nPredicting for product:")
        print(f"  Designation: {args.designation}")
        print(f"  Description: {args.description}")
        result = predictor.predict_product(
            args.designation,
            args.description,
            return_probabilities=args.probabilities,
            top_k=args.top_k,
        )
        print("\nResult:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.file:
        # Batch prediction from file
        print(f"\nLoading texts from {args.file}...")
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different file formats
        if isinstance(data, list):
            texts = data
        elif isinstance(data, dict) and "texts" in data:
            texts = data["texts"]
        else:
            texts = [data]

        print(f"Found {len(texts)} texts")

        results = predictor.predict_batch(
            texts,
            batch_size=args.batch_size,
            return_probabilities=args.probabilities,
            top_k=args.top_k,
        )

        # Save results
        output_file = args.file.replace(".json", "_predictions.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n Results saved to {output_file}")

        # Print summary
        print("\nSummary:")
        print(f"  Total predictions: {len(results)}")
        categories = [r["predicted_category"] for r in results]
        unique_cats = len(set(categories))
        print(f"  Unique categories predicted: {unique_cats}")
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        print(f"  Average confidence: {avg_confidence:.3f}")

    else:
        print(" Error: Provide --text, --designation, or --file argument")

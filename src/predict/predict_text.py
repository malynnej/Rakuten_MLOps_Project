# src/models/predict.py

import torch
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Union, List, Dict
import numpy as np


class TextPredictor:
    """Production prediction class for API endpoints"""
    
    def __init__(self, model_path="./models/bert-rakuten-final"):
        self.model_path = model_path
        
        # Load model and tokenizer
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
        
        print(f"Predictor ready - Model: {model_path}, Device: {self.device}")
    
    def predict(self, text: Union[str, List[str]], return_probabilities: bool = False) -> Union[Dict, List[Dict]]:
        """
        Predict category for single text or batch of texts.
        
        Args:
            text: Single text string or list of texts
            return_probabilities: If True, return top-5 probabilities
        
        Returns:
            Single dict or list of dicts with predictions
        """
        # Handle single text
        single_input = isinstance(text, str)
        if single_input:
            text = [text]
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get predictions
        pred_indices = torch.argmax(logits, dim=-1).cpu().numpy()
        confidences = probs.max(dim=-1).values.cpu().numpy()
        
        # Convert to original categories
        original_categories = self.le.inverse_transform(pred_indices)
        
        # Build results
        results = []
        for i, (cat, conf, pred_idx) in enumerate(zip(original_categories, confidences, pred_indices)):
            result = {
                "text": text[i],
                "predicted_category": int(cat),
                "confidence": float(conf),
                "predicted_index": int(pred_idx)
            }
            
            # Add top-5 probabilities if requested
            if return_probabilities:
                top5_prob, top5_idx = torch.topk(probs[i], k=min(5, len(probs[i])))
                result["top_5_predictions"] = [
                    {
                        "category": int(self.le.inverse_transform([int(idx)])[0]),
                        "probability": float(prob)
                    }
                    for prob, idx in zip(top5_prob.cpu().numpy(), top5_idx.cpu().numpy())
                ]
            
            results.append(result)
        
        # Return single dict if single input
        return results[0] if single_input else results
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Predict for large batches efficiently"""
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = self.predict(batch)
            all_results.extend(results if isinstance(results, list) else [results])
        
        return all_results


# CLI usage
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Predict with BERT model")
    parser.add_argument("--model_path", type=str, default="./models/bert-rakuten-final")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--file", type=str, help="JSON file with texts to classify")
    parser.add_argument("--top5", action="store_true", help="Return top-5 predictions")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = TextPredictor(args.model_path)
    
    # Predict
    if args.text:
        result = predictor.predict(args.text, return_probabilities=args.top5)
        print(json.dumps(result, indent=2))
    
    elif args.file:
        with open(args.file, 'r') as f:
            data = json.load(f)
        texts = data if isinstance(data, list) else [data]
        results = predictor.predict_batch(texts)
        print(json.dumps(results, indent=2))
    
    else:
        print("Provide --text or --file argument")

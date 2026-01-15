################################################
### IMPORT LIBRARIES 
################################################

import pandas as pd
import numpy as np

import torch
import pickle
import json
import argparse

from datasets import load_dataset
from datasets import Dataset
from datasets import DatasetDict

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback

from evaluate import load
import os
from pathlib import Path

def train_bert_model(retrain=False, model_path="./models/bert-rakuten-final"):
    
    """
    Train or retrain BERT text classification model with LLRD and advanced optimizations.
    
    Args:
        retrain (bool): If True, loads existing label encoder and model config.
                       If False, creates new label encoder (initial training).
        model_path (str): Path to save/load model artifacts
    
    Returns:
        dict: Training metrics and model info
    """
    
    print(f"{'='*50}\n{'RETRAINING' if retrain else 'INITIAL TRAINING'} MODE\n{'='*50}")

    ################################################
    ### SETTINGS 
    ################################################

    # show working directory
    print("Working directory:", os.getcwd())

    # define device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")


    ################################################
    ### DATA PREPARATION 
    ################################################

    # Load preprocessed data from pipeline
    X_train = pd.read_csv("./src/data/preprocessed/X_train.csv", index_col=0)
    y_train = pd.read_csv("./src/data/preprocessed/y_train.csv", index_col=0)
    X_test = pd.read_csv("./src/data/preprocessed/X_test.csv", index_col=0)
    y_test = pd.read_csv("./src/data/preprocessed/y_test.csv", index_col=0)

    # Handle label encoding
    le_path = os.path.join(model_path, "label_encoder.pkl")
    
    if retrain:
        # RETRAINING: Load existing label encoder
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"Label encoder not found at {le_path}. Cannot retrain without existing encoder!")
        
        with open(le_path, "rb") as f:
            le = pickle.load(f)
        print(f"Loaded existing label encoder with {len(le.classes_)} classes")
        print(f"   Classes: {le.classes_}")
        
        # Transform with existing encoder (will fail if new categories appear)
        try:
            y_train_encoded = le.transform(y_train.values.ravel())
            y_test_encoded = le.transform(y_test.values.ravel())
        except ValueError as e:
            raise ValueError(f"New categories detected in data! Existing encoder doesn't support them: {e}")
    else:
        # INITIAL TRAINING: Create new label encoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train.values.ravel())
        y_test_encoded = le.transform(y_test.values.ravel())
        print(f"Created new label encoder with {len(le.classes_)} classes")
        print(f"   Classes: {le.classes_}")
        
        # Save label encoder
        os.makedirs(model_path, exist_ok=True)
        with open(le_path, "wb") as f:
            pickle.dump(le, f)
        print(f"Saved label encoder to {le_path}")



    # Create id2label and label2id from the encoder
    id2label = {i: str(cat_id) for i, cat_id in enumerate(le.classes_)}
    label2id = {str(cat_id): i for i, cat_id in enumerate(le.classes_)}

    if not retrain:
        # Save label mapping
        with open(os.path.join(model_path,"id2label.json"), "w") as f:
            json.dump(id2label, f)
        print(f"Saved label encoder json to {model_path}")
    
    num_labels = len(le.classes_)

    print("="*50)
    print("LABEL MAPPINGS")
    print("="*50)
    print(f"Total classes: {len(le.classes_)}")
    print(f"\nOriginal category IDs (sorted): {le.classes_}")
    print(f"\nid2label (first 5): {dict(list(id2label.items())[:5])}")
    print(f"label2id (first 5): {dict(list(label2id.items())[:5])}")

    # Split train into train/val (80/20)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
    )

    # Combine into datasets
    train_df = pd.DataFrame({"text": X_train_split["text"].values, "labels": y_train_split})
    val_df = pd.DataFrame({"text": X_val_split["text"].values, "labels": y_val_split})
    test_df = pd.DataFrame({"text": X_test["text"].values, "labels": y_test_encoded})

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Convert pandas to HF Dataset
    dataset_train = Dataset.from_pandas(train_df)
    dataset_val = Dataset.from_pandas(val_df)
    dataset_test = Dataset.from_pandas(test_df)

    # Combine into DatasetDict
    dataset = DatasetDict({
        "train": dataset_train,
        "val": dataset_val,
        "test": dataset_test
    })

    print(dataset)

    ################################################
    ### TOKENIZER SETUP AND TOKENIZATION
    ################################################

    # Load tokenizer
    if retrain and os.path.exists(model_path):
        # Load existing tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Loaded existing tokenizer from {model_path}")
    else:
        # Load base tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("Loaded base BERT tokenizer")

    # Define tokenize function
    def tokenize_function(examples):

        texts = [
            text if (text is not None and text.strip() != "") else "[EMPTY]"
            for text in examples["text"]
        ]
        
        #return tokenizer(examples["text"],
        return tokenizer(texts,
                        #padding="max_length",   #removed since using DataCollator
                        truncation=True,
                        max_length=128)

    # tokenize dataset into batches
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets["test"].save_to_disk("./src/data/processed/test_dataset")

    # Initialize `DataCollatorWithPadding` to handle dynamic padding during training.
    data_collator = DataCollatorWithPadding(tokenizer)

    ################################################
    ### MODEL SETUP
    ################################################

    # Load model
    if retrain and os.path.exists(os.path.join(model_path, "config.json")):
        # Load existing model for fine-tuning
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        print(f"Loaded existing model from {model_path} for retraining")
    else:
        # Load fresh base model
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        print("Loaded fresh pretrained BERT base model")

    # Freeze entire BERT
    for p in model.bert.parameters():
        p.requires_grad = False

    # Unfreeze last two transformer layers
    for name, param in model.bert.named_parameters():
        if name.startswith("encoder.layer.10") or name.startswith("encoder.layer.11"):
            param.requires_grad = True

    # Unfreeze pooler (optional but recommended)
    for name, param in model.bert.named_parameters():
        if "pooler" in name:
            param.requires_grad = True

    # Unfreeze classifier head
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Move model to device 
    model.to(device)
    print(f"Model moved to: {next(model.parameters()).device}")

    ###########################################################
    ### SETUP LAYER-WISE LEARNING RATE DECAY (LLRD) AND TRAINER
    ###########################################################

    # Metrics 
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": np.mean(preds == labels)}

    # Define LLRD parameter groups
    def get_llrd_params(model, base_lr=5e-5):
        # Layer-wise learning rate decay
        lr_factor = 0.85

        opt_params = []
        
        # Classifier head (highest LR)
        opt_params.append({
            "params": model.classifier.parameters(),
            "lr": base_lr * 3.0
        })

        # Pooler (optional)
        opt_params.append({
            "params": model.bert.pooler.parameters(),
            "lr": base_lr
        })

        # Encoder layers: progressively smaller LR
        for layer_idx in range(11, -1, -1):  # layer 11 down → 0
            layer = f"encoder.layer.{layer_idx}."
            params = [p for n, p in model.bert.named_parameters() if n.startswith(layer)]

            if len(params) == 0:
                continue

            layer_lr = base_lr * (lr_factor ** (11 - layer_idx))

            opt_params.append({
                "params": params,
                "lr": layer_lr
            })

        # Embeddings (smallest LR)
        embed_params = list(model.bert.embeddings.parameters())
        opt_params.append({
            "params": embed_params,
            "lr": base_lr * (lr_factor ** 12)
        })

        return opt_params

    # Training Arguments 
    training_args = TrainingArguments(
        output_dir=os.path.join(model_path, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        #learning_rate=5e-5,             # Base LR (LLRD scales from this)
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        load_best_model_at_end=True,    # keep best model according to val accuracy
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,                      # Or bf16=True
        report_to="none"
    )

    optimizer_params = get_llrd_params(model, base_lr=5e-5)

    # Trainer Initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(
            torch.optim.AdamW(
                optimizer_params,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01
            ),
            None  # HF Trainer will create scheduler automatically
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    )

    # Verify trainer device
    print(f"Trainer using: {trainer.args.device}")

    ###########################################################
    ### TRAINING MODEL
    ###########################################################

    # Training model
    train_result = trainer.train()

    # Evaluate on test set
    test_dataset = tokenized_datasets["test"]
    test_metrics = trainer.evaluate(test_dataset)
    print(f"\n{'='*50}\nTEST SET RESULTS\n{'='*50}")
    print(f"Test Accuracy: {test_metrics['eval_accuracy']:.4f}")
    print(f"Test Loss: {test_metrics['eval_loss']:.4f}")

    # Save model
    trainer.save_model(model_path)

    print(f"Model saved to {model_path}")

    # Return metrics
    return {
        "train_loss": train_result.training_loss,
        "eval_accuracy": test_metrics["eval_accuracy"],
        "eval_loss": test_metrics["eval_loss"],
        "num_labels": num_labels,
        "trainable_params": trainable_params,
        "model_path": model_path
    }

###########################################################
### EXECUTION
###########################################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train BERT text classification model")
    parser.add_argument("--retrain", action="store_true", help="Retrain existing model")
    parser.add_argument("--model_path", type=str, default="./models/bert-rakuten-final", 
                       help="Path to save/load model")
    
    args = parser.parse_args()
    
    # Train
    metrics = train_bert_model(retrain=args.retrain, model_path=args.model_path)
    
    print(f"\n{'='*50}\nTRAINING COMPLETE\n{'='*50}")
    print(f"Final metrics: {metrics}")
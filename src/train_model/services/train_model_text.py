# src/train_model/services/train_model_text.py
"""
BERT Text Classification Training Pipeline

Loads preprocessed parquet data from Data Service.
Trains BERT model with Layer-wise Learning Rate Decay (LLRD).
"""

import argparse
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from core.config import get_path, load_config
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def train_bert_model(retrain: bool = False, model_name: str = "bert-rakuten-final"):
    """
    Train or retrain BERT text classification model with LLRD and advanced optimizations.

    Args:
        retrain (bool): If True, loads existing model for fine-tuning.
                       If False, trains from pretrained BERT base (initial training).
        model_name (str): Name of model directory to save/load

    Returns:
        tuple: (trainer, metrics_dict)
    """

    print(f"\n{'=' * 60}")
    print(f"{'RETRAINING' if retrain else 'INITIAL TRAINING'} MODE")
    print(f"{'=' * 60}\n")

    # Load configuration
    params = load_config("params")

    # Extract training parameters
    train_params = params["training"]

    # Use model_name from params if not provided
    if model_name is None:
        model_name = train_params["model_name"]

    ################################################
    ### DEVICE SETUP
    ################################################

    # Device selection (same as prediction service)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(" Using Apple Metal (MPS) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f" Using CUDA acceleration (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(" Using CPU (consider GPU for faster training)")

    print(f"Working directory: {os.getcwd()}")

    ################################################
    ### LOAD PREPROCESSED DATA FROM PARQUET
    ################################################

    print(f"\n{'=' * 60}")
    print("LOADING PREPROCESSED DATA")
    print(f"{'=' * 60}\n")

    # Get paths from config
    preprocessed_dir = get_path("data.preprocessed")

    # Load parquet files created by Data Service (tokenized already)
    print("Loading parquet files from Data Service...")
    train_df = pd.read_parquet(preprocessed_dir / "train.parquet")
    val_df = pd.read_parquet(preprocessed_dir / "val.parquet")
    test_df = pd.read_parquet(preprocessed_dir / "test.parquet")

    print(f"✓ Train: {len(train_df):,} samples")
    print(f"✓ Val:   {len(val_df):,} samples")
    print(f"✓ Test:  {len(test_df):,} samples")

    # Verify required columns exist
    required_cols = ["text", "input_ids", "attention_mask", "labels", "prdtypecode"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Required column '{col}' not found in training data")

    print(f"\nColumns available: {list(train_df.columns)}")
    print(f"Sample text: {train_df['text'].iloc[0][:100]}...")

    ################################################
    ### LOAD LABEL ENCODER (Created by Data Service)
    ################################################

    print(f"\n{'=' * 60}")
    print("LOADING LABEL ENCODER")
    print(f"{'=' * 60}\n")

    # Load label encoder created by Data Service
    models_dir = get_path("models.save_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    le_path = models_dir / "label_encoder.pkl"
    mappings_path = models_dir / "label_mappings.json"

    if not le_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found at {le_path}!\n"
            f"Run Data Service preprocessing first:\n"
            f"  curl -X POST http://localhost:8001/preprocess/from-raw"
        )

    # Load label encoder
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    print(f"✓ Loaded label encoder with {len(le.classes_)} classes")
    print(f"  Category range: {le.classes_.min()} - {le.classes_.max()}")

    # Load label mappings
    with open(mappings_path, "r") as f:
        mappings = json.load(f)

    id2label = {int(k): int(v) for k, v in mappings["id2label"].items()}
    label2id = {int(k): int(v) for k, v in mappings["label2id"].items()}
    num_labels = mappings["num_labels"]

    print("\nLabel mappings loaded:")
    print(f"  Total classes: {num_labels}")
    print(f"  Sample id2label: {dict(list(id2label.items())[:5])}")

    ################################################
    ### CREATE HUGGINGFACE DATASETS
    ################################################

    print(f"\n{'=' * 60}")
    print("CREATING HUGGINGFACE DATASETS")
    print(f"{'=' * 60}\n")

    # Convert pandas DataFrames to HuggingFace Datasets
    # Note: Data is already tokenized by Data Service!
    dataset_train = Dataset.from_pandas(train_df[["input_ids", "attention_mask", "labels"]])
    dataset_val = Dataset.from_pandas(val_df[["input_ids", "attention_mask", "labels"]])
    dataset_test = Dataset.from_pandas(test_df[["input_ids", "attention_mask", "labels"]])

    # Combine into DatasetDict
    dataset = DatasetDict({"train": dataset_train, "val": dataset_val, "test": dataset_test})

    print("✓ Created HuggingFace datasets:")
    print(dataset)

    # Verify data types (input_ids and attention_mask should be lists)
    print("\nData types:")
    print(f"  input_ids type: {type(dataset_train['input_ids'][0])}")
    print(f"  attention_mask type: {type(dataset_train['attention_mask'][0])}")
    print(f"  labels type: {type(dataset_train['labels'][0])}")

    ################################################
    ### TOKENIZER SETUP
    ################################################

    print(f"\n{'=' * 60}")
    print("LOADING TOKENIZER")
    print(f"{'=' * 60}\n")

    # Model save path
    model_path = models_dir / model_name

    # Load tokenizer
    bert_model = train_params["base_model"]

    if retrain and model_path.exists():
        # Load existing tokenizer from saved model
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        print(f" Loaded existing tokenizer from {model_path}")
    else:
        # Load base tokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        print(f" Loaded base tokenizer: {bert_model}")

    # Data collator (handles dynamic padding during training)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ################################################
    ### MODEL SETUP
    ################################################

    print(f"\n{'=' * 60}")
    print("LOADING MODEL")
    print(f"{'=' * 60}\n")

    # Load model
    if retrain and model_path.exists():
        # Load existing model for fine-tuning
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        print(f" Loaded existing model from {model_path} for retraining")
    else:
        # Load fresh pretrained base model
        model = AutoModelForSequenceClassification.from_pretrained(
            bert_model, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        print(f" Loaded fresh pretrained model: {bert_model}")

    ################################################
    ### FREEZE/UNFREEZE LAYERS (LLRD Setup)
    ################################################

    print(f"\n{'=' * 60}")
    print("CONFIGURING LAYER FREEZING")
    print(f"{'=' * 60}\n")

    # Freeze entire BERT base
    for p in model.bert.parameters():
        p.requires_grad = False

    # Unfreeze specific encoder layers from config
    unfreeze_layers = train_params["unfreeze_encoder_layers"]
    for layer_idx in unfreeze_layers:
        for name, param in model.bert.named_parameters():
            if name.startswith(f"encoder.layer.{layer_idx}"):
                param.requires_grad = True
    print(f" Unfroze encoder layers: {unfreeze_layers}")

    # Unfreeze pooler if configured
    if train_params["unfreeze_pooler"]:
        for name, param in model.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
        print(" Unfroze pooler")

    # Unfreeze classifier (always)
    for p in model.classifier.parameters():
        p.requires_grad = True
    print(" Unfroze classifier")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_params / total_params

    print(f" Total parameters:     {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,} ({trainable_pct:.1f}%)")
    print(f" Frozen parameters:    {total_params - trainable_params:,}")

    # Move model to device
    model.to(device)
    print(f"\n Model moved to: {device}")

    ################################################
    ### LAYER-WISE LEARNING RATE DECAY (LLRD)
    ################################################

    def get_llrd_params(model, base_lr=5e-5, lr_decay=0.85):
        """
        Create parameter groups with layer-wise learning rate decay.

        Strategy:
        - Classifier head: base_lr * 3.0 (highest LR)
        - Pooler: base_lr
        - Encoder layers: progressively smaller (layer 11 → 0)
        - Embeddings: smallest LR

        Args:
            model: BERT model
            base_lr: Base learning rate
            lr_decay: Decay factor between layers (0.85 = 15% decay per layer)

        Returns:
            list: Parameter groups for optimizer
        """
        base_lr = train_params["learning_rate"]
        lr_decay = train_params["llrd"]["lr_decay_factor"]
        classifier_mult = train_params["llrd"]["classifier_lr_multiplier"]

        opt_params = []

        # Classifier head (highest LR)
        opt_params.append(
            {"params": model.classifier.parameters(), "lr": base_lr * classifier_mult}
        )

        # Pooler
        if train_params["unfreeze_pooler"]:
            opt_params.append({"params": model.bert.pooler.parameters(), "lr": base_lr})

        # Encoder layers
        for layer_idx in range(11, -1, -1):
            if layer_idx not in train_params["unfreeze_encoder_layers"]:
                continue

            layer_name = f"encoder.layer.{layer_idx}."
            layer_params = [
                p
                for n, p in model.bert.named_parameters()
                if n.startswith(layer_name) and p.requires_grad
            ]

            if len(layer_params) == 0:
                continue

            layer_lr = base_lr * (lr_decay ** (11 - layer_idx))
            opt_params.append({"params": layer_params, "lr": layer_lr})

        return opt_params

    ################################################
    ### METRICS COMPUTATION
    ################################################

    def compute_metrics(eval_pred):
        """Compute accuracy metric for evaluation"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        return {"accuracy": accuracy}

    ################################################
    ### TRAINING ARGUMENTS
    ################################################

    print(f"\n{'=' * 60}")
    print("CONFIGURING TRAINING")
    print(f"{'=' * 60}\n")

    # Create checkpoints directory
    checkpoints_dir = model_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        #  From config
        num_train_epochs=train_params["num_train_epochs"],
        per_device_train_batch_size=train_params["per_device_train_batch_size"],
        per_device_eval_batch_size=train_params["per_device_eval_batch_size"],
        warmup_ratio=train_params["warmup_ratio"],
        weight_decay=train_params["weight_decay"],
        eval_strategy=train_params["eval_strategy"],
        save_strategy=train_params["save_strategy"],
        logging_strategy=train_params["logging_strategy"],
        load_best_model_at_end=train_params["load_best_model_at_end"],
        metric_for_best_model=train_params["metric_for_best_model"],
        greater_is_better=train_params["early_stopping"]["greater_is_better"],
        save_total_limit=train_params["save_total_limit"],
        logging_first_step=train_params["logging_first_step"],
        fp16=train_params["fp16"] and torch.cuda.is_available(),
        dataloader_num_workers=train_params["dataloader_num_workers"],
        gradient_accumulation_steps=train_params["gradient_accumulation_steps"],
        max_grad_norm=train_params["max_grad_norm"],
        report_to=train_params["report_to"],
    )

    print("Training configuration:")
    print(f"  Batch size:     {training_args.per_device_train_batch_size}")
    print(f"  Epochs:         {training_args.num_train_epochs}")
    print(f"  Warmup ratio:   {training_args.warmup_ratio}")
    print(f"  Weight decay:   {training_args.weight_decay}")
    print(f"  FP16:           {training_args.fp16}")
    print(f"  Device:         {training_args.device}")

    ################################################
    ###  OPTIMIZER FROM CONFIG
    ################################################

    if train_params["llrd"]["enabled"]:
        optimizer_params = get_llrd_params(model, train_params)
        print(f"\n Using LLRD with {len(optimizer_params)} parameter groups")
    else:
        optimizer_params = model.parameters()
        print("\n Using standard optimizer (no LLRD)")

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=train_params["learning_rate"],  # Base LR (LLRD will override per group)
        betas=(train_params["adam_beta1"], train_params["adam_beta2"]),
        eps=train_params["adam_epsilon"],
        weight_decay=train_params["weight_decay"],
    )

    ################################################
    ### TRAINER INITIALIZATION
    ################################################

    print(f"\n{'=' * 60}")
    print("INITIALIZING TRAINER")
    print(f"{'=' * 60}\n")

    callbacks = []
    if train_params["early_stopping"]["enabled"]:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_params["early_stopping"]["patience"]
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        # tokenizer=tokenizer,   #not supported furthermore
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )

    print("✓ Trainer initialized")
    print(f"  Training samples:   {len(dataset['train']):,}")
    print(f"  Validation samples: {len(dataset['val']):,}")
    print(f"  Test samples:       {len(dataset['test']):,}")

    ################################################
    ### TRAINING
    ################################################

    print(f"\n{'=' * 60}")
    print("STARTING TRAINING")
    print(f"{'=' * 60}\n")

    start_time = datetime.now()

    # Train model
    train_result = trainer.train()

    end_time = datetime.now()
    training_duration = end_time - start_time

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}\n")
    print(f"Training duration: {training_duration}")
    print(f"Final training loss: {train_result.training_loss:.4f}")

    ################################################
    ### EVALUATION ON TEST SET
    ################################################

    print(f"\n{'=' * 60}")
    print("EVALUATING ON TEST SET")
    print(f"{'=' * 60}\n")

    test_metrics = trainer.evaluate(dataset["test"])

    print("Test Results:")
    print(f"  Accuracy: {test_metrics['eval_accuracy']:.4f}")
    print(f"  Loss:     {test_metrics['eval_loss']:.4f}")

    ################################################
    ### SAVE MODEL AND ARTIFACTS
    ################################################

    print(f"\n{'=' * 60}")
    print("SAVING MODEL")
    print(f"{'=' * 60}\n")

    # Save model
    trainer.save_model(str(model_path))
    print(f"✓ Model saved to: {model_path}")

    # Delete checkpoints
    for checkpoint in Path(model_path).glob("checkpoints/checkpoint-*"):
        print(f"Deleting {checkpoint.name}")
        shutil.rmtree(checkpoint)

    # Save training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "mode": "retrain" if retrain else "initial",
        "num_labels": num_labels,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["val"]),
        "test_samples": len(dataset["test"]),
        "training_duration_seconds": training_duration.total_seconds(),
        "final_train_loss": float(train_result.training_loss),
        "test_accuracy": float(test_metrics["eval_accuracy"]),
        "test_loss": float(test_metrics["eval_loss"]),
        "device": str(device),
        "base_model": bert_model,
        "max_length": params["preprocessing"]["max_length"],
    }

    metadata_path = model_path / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f" Training metadata saved to: {metadata_path}")

    # Save training/validation comparison
    comparison = {
        "train_val_comparison": {
            "train_loss": float(train_result.training_loss),
            "val_accuracy": float(test_metrics["eval_accuracy"]),
            "val_loss": float(test_metrics["eval_loss"]),
        }
    }

    evaluation_dir = get_path("results.evaluation") / model_name
    comparison_path = evaluation_dir / "train_val_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f" Train/val comparison saved to: {comparison_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}\n")
    print(f"Model:           {model_name}")
    print(f"Mode:            {'Retraining' if retrain else 'Initial training'}")
    print(f"Duration:        {training_duration}")
    print(f"Device:          {device}")
    print(f"Train loss:      {train_result.training_loss:.4f}")
    print(f"Test accuracy:   {test_metrics['eval_accuracy']:.4f}")
    print(f"Test loss:       {test_metrics['eval_loss']:.4f}")
    print(f"Saved to:        {model_path}")

    return trainer, metadata


################################################
### CLI EXECUTION
################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BERT text classification model for Rakuten products"
    )
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain existing model (default: initial training)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-rakuten-final",
        help="Name of model directory (default: bert-rakuten-final)",
    )

    args = parser.parse_args()

    # Train
    try:
        trainer, metrics = train_bert_model(retrain=args.retrain, model_name=args.model_name)

        print("\n" + "=" * 60)
        print(" SUCCESS: Training completed successfully!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(" ERROR: Training failed!")
        print("=" * 60)
        print(f"\nError details: {e}")
        raise

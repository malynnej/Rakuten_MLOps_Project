"""
Convert holdout_raw.parquet to JSON for prediction testing.

Extracts designation and description fields (raw data) 
and saves as JSON for easy prediction testing.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))



def create_holdout_json(
    num_samples: int = 100,
    output_name: str = "holdout_test_samples.json"
):
    """
    Convert holdout parquet to JSON format.
    
    Args:
        num_samples: Number of samples to extract (default: 100)
        output_name: Output JSON filename
    """
    print("="*60)
    print("CREATING HOLDOUT TEST JSON")
    print("="*60)
    
    # Load holdout data
    holdout_path = Path("data/preprocessed/holdout_raw.parquet")
    
    if not holdout_path.exists():
        print(f" Holdout data not found at: {holdout_path}")
        print("   Run preprocessing first:")
        print("   uv run --package data python src/data/services/preprocess/text_preparation_pipeline.py")
        return
    
    print(f"\n✓ Loading holdout data from: {holdout_path}")
    holdout_df = pd.read_parquet(holdout_path)
    
    print(f"✓ Loaded {len(holdout_df)} samples")
    print(f"  Columns: {list(holdout_df.columns)}")
    
    # Sample data
    if num_samples < len(holdout_df):
        sample_df = holdout_df.sample(n=num_samples, random_state=42)
        print(f"✓ Sampled {num_samples} random samples")
    else:
        sample_df = holdout_df
        print(f"✓ Using all {len(sample_df)} samples")
    
    # Create JSON structure
    # Format 1: Simple text list (for basic testing)
    texts = sample_df['text'].tolist()
    
    # Format 2: Full product data (with ground truth)
    products = []
    for _, row in sample_df.iterrows():
        product = {
            "designation": row.get('designation', ''),
            "description": row.get('description', ''),
            "text": row['text'],
            "true_category": int(row['prdtypecode']),
            "true_label": int(row['labels'])
        }
        products.append(product)
    
    # Save to data directory (accessible from anywhere)
    data_dir = Path('data') # Go up to data/ directory
    output_dir = data_dir / "test_samples"
    output_dir.mkdir(exist_ok=True)
    
    # Save simple format
    simple_path = output_dir / f"simple_{output_name}"
    with open(simple_path, 'w', encoding='utf-8') as f:
        json.dump({"texts": texts}, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved simple format to: {simple_path}")
    print("  Format: {\"texts\": [\"text1\", \"text2\", ...]}")
    
    # Save full format
    full_path = output_dir / f"full_{output_name}"
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump({"products": products}, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved full format to: {full_path}")
    print("  Format: {\"products\": [{designation, description, text, true_category, ...}, ...]}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(sample_df)}")
    print(f"Unique categories: {sample_df['prdtypecode'].nunique()}")
    print(f"Output location: {output_dir}")
    
    # Show sample
    print(f"\n{'='*60}")
    print("SAMPLE DATA (first 3)")
    print(f"{'='*60}")
    for i, product in enumerate(products[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Designation: {product['designation'][:60]}")
        print(f"  Description: {product['description'][:60]}")
        print(f"  Category: {product['true_category']}")
    
    print(f"\n{'='*60}")
    print("USAGE EXAMPLES")
    print(f"{'='*60}")
    print("\n1. Test with simple format:")
    print("   uv run --package predict python src/predict/services/predict_text.py \\")
    print(f"     --file {simple_path}")
    
    print("\n2. Test with full format (use in custom script):")
    print(f"   python scripts/test_predictions.py --file {full_path}")
    
    return simple_path, full_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert holdout data to JSON for prediction testing"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to extract (default: 100)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="holdout_test_samples.json",
        help="Output JSON filename (default: holdout_test_samples.json)"
    )
    
    args = parser.parse_args()
    
    create_holdout_json(
        num_samples=args.num_samples,
        output_name=args.output_name
    )

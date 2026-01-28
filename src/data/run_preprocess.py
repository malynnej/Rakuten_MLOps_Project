from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from core.config import get_path
from services.preprocess.text_preparation_pipeline import TextPreparationPipeline


def main() -> None:
    """
    DVC entrypoint:
    """

    from urllib.request import urlretrieve

    from core.config import load_config

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    bucket = load_config("paths")["data"]["bucket_raw"].rstrip("/") + "/"

    x_name = Path(get_path("data.X_train_raw")).name
    y_name = Path(get_path("data.y_train_raw")).name

    x_path = raw_dir / x_name
    y_path = raw_dir / y_name

    if not x_path.exists():
        print(f"Downloading {x_name} ...")
        urlretrieve(bucket + x_name, x_path)

    if not y_path.exists():
        print(f"Downloading {y_name} ...")
        urlretrieve(bucket + y_name, y_path)

    print(f"✓ Raw files ready:\n- {x_path}\n- {y_path}")

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    label_col = "prdtypecode" if "prdtypecode" in y.columns else y.columns[0]
    df = X.copy()
    df[label_col] = y[label_col].values

    pipe = TextPreparationPipeline()
    pipe.prepare_training_data(df, combine_existing_data=False, save_holdout=True)


if __name__ == "__main__":
    main()

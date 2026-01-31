import logging
from pathlib import Path

import requests
from core.config import get_path, load_config


def import_raw_data(raw_data_path, filenames, bucket_url):
    """import filenames from bucket_folder_url in raw_data_path"""
    raw_data_path = Path(raw_data_path)
    raw_data_path.mkdir(parents=True, exist_ok=True)

    # download all the files
    for filename in filenames:
        object_url = f"{bucket_url}{filename}"
        output_file = raw_data_path / filename
        if not output_file.exists():
            print(f"downloading {filename}...")
            response = requests.get(object_url)
            if response.status_code == 200:
                # Process the response content as needed
                content = (
                    response.content
                )  
                with open(output_file, "wb") as file:
                    file.write(content)
            else:
                print(f"Error accessing the object {filename}:", response.status_code)

def main():
    """Download data to raw data folder"""
    # Use get_path helper for cleaner code
    raw_dir = get_path("data.raw")
    
    paths = load_config("paths")
    bucket_url = paths["data"]["bucket_raw"]
    filenames = [
        paths["data"]["X_train_raw"],
        paths["data"]["y_train_raw"]
    ]
    
    import_raw_data(raw_dir, filenames, bucket_url)
    
    logger = logging.getLogger(__name__)
    logger.info("Raw data import complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
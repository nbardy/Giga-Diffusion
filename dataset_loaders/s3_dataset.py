###
# Streaming S3 dataset loader
###
import boto3
import json
import re
import os
from torch.utils.data import Dataset, DataLoader

import hashlib
import pickle


import datetime

from urllib.parse import urlparse


data_loader_log = "dataloader.log"

DEFAULT_CACHE_DIR = "/tmp/s3_cache"


class S3StreamingJSONDataset(Dataset):
    def __init__(
        self,
        s3_bucket=None,
        s3_folder=None,
        filename_pattern=None,
        s3_url=None,
        offset=0,
        cache_dir=DEFAULT_CACHE_DIR,
    ):
        if s3_url:
            self.s3_url = s3_url
            parsed_url = urlparse(s3_url)

            if not parsed_url.netloc or not parsed_url.path:
                raise ValueError("Invalid S3 URL")

            self.bucket = parsed_url.netloc
            self.folder = parsed_url.path.lstrip("/")

            log_to_file(f"Bucket: {self.bucket}", data_loader_log)
            log_to_file(f"Folder: {self.folder}", data_loader_log)
        else:
            self.bucket = s3_bucket
            self.folder = s3_folder

        self.filename_pattern = filename_pattern
        self.offset = offset
        self.marker = None
        self.files = []

        # Set cache
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.total_files = 0
        while True:
            self.load_next_page()
            if not self.marker:
                break
        self.total_files = len(self.files)

    def __len__(self):
        return self.total_files

    def __getitem__(self, idx):
        log_to_file(f"__getitem__ {idx}", data_loader_log)
        data_idx, item = self.fetch_item(idx)
        return (data_idx, item) if item is not None else None

    def list_objects_with_cache(self, params):
        # Create a hash of the params
        cache_key = hashlib.md5(str(params).encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        # If the cache file exists, load and return the data
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Otherwise, fetch the data, cache it, and return it
        s3 = boto3.client("s3")
        response = s3.list_objects(**params)
        with open(cache_path, "wb") as f:
            pickle.dump(response, f)

        return response

    def load_next_page(self):
        params = {
            "Bucket": self.bucket,
            "Prefix": self.folder,
            "MaxKeys": 1000,
        }

        if self.marker:
            params["Marker"] = self.marker

        resp = self.list_objects_with_cache(params)

        if "Contents" in resp:
            for item in resp["Contents"]:
                if self.filename_pattern is None:
                    self.files.append(item["Key"])
                elif re.search(self.filename_pattern, item["Key"]):
                    self.files.append(item["Key"])

        self.marker = resp.get("NextMarker")

    def s3_get_object_with_cache(self, bucket, key):
        """
        Fetches an S3 object with caching. If the object is already cached, it is loaded from the cache.
        Otherwise, it is fetched from S3 and then cached for future use.
        """
        s3 = boto3.client("s3")
        cache_path = os.path.join(self.cache_dir, key.replace("/", "_"))

        # If the cache file exists, load and return the data
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

        # Otherwise, fetch the data, cache it, and return it
        try:
            log_to_file(f"Fetching file: s3://{bucket}/{key}", data_loader_log)
            response = s3.get_object(Bucket=bucket, Key=key)
            text_data = response["Body"].read().decode("utf-8")
            json_data = json.loads(text_data)
            with open(cache_path, "w") as f:
                json.dump(json_data, f)
            return json_data
        except Exception as e:
            log_to_file(f"Skipping file due to error: {e}", data_loader_log)
            return None

    def fetch_item(self, idx):
        """
        Fetches an item from the dataset at the given index. The item is fetched from cache if available,
        otherwise it is fetched from S3 and then cached for future use.
        """
        s3_file = self.files[idx]

        match = re.search(r"(\d+)_neighbors.json", s3_file)
        if match:
            data_idx = int(match.group(1))
        else:
            raise ValueError(f"Invalid filename pattern: {s3_file}")

        json_data = self.s3_get_object_with_cache(self.bucket, s3_file)
        if json_data is None:
            raise ValueError(f"data_idx: {data_idx}")

        return data_idx, json_data


class CustomCollate:
    def __call__(self, batch):
        flattened = [item for item in batch if item is not None]
        return flattened if len(flattened) > 0 else None


def log_to_file(message, log_file="main_log", worker_id=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = (
        f"[{timestamp}] [Worker {worker_id if worker_id else 'Main'}]: {message}"
    )

    with open(log_file, "a") as f:
        f.write(full_message + "\n")

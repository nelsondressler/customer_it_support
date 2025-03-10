import os
import sys

def add_prefix_path(file_path: str, prefix_path: str):
    return os.path.join(prefix_path, file_path)
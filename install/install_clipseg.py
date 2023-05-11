#install/install_clipseg.py
import os
import requests
from zipfile import ZipFile
from io import BytesIO

def install():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    clipseg_models_directory = os.path.join(base_directory, "..", "..", "..", "models", "clipseg")

    if not os.path.exists(clipseg_models_directory):
        print(f"{clipseg_models_directory} does not exist. Creating.")
        os.makedirs(clipseg_models_directory)

    files_to_download = {
        "clipseg.py": "https://raw.githubusercontent.com/timojl/clipseg/master/models/clipseg.py",
        "vitseg.py": "https://raw.githubusercontent.com/timojl/clipseg/master/models/vitseg.py",
    }

    for file_name, file_url in files_to_download.items():
        file_path = os.path.join(clipseg_models_directory, file_name)
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            response = requests.get(file_url)
            with open(file_path, "wb") as f:
                f.write(response.content)

    weights_url = "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download"
    weights_zip_file_path = os.path.join(clipseg_models_directory, "weights.zip")

    if not os.path.exists(weights_zip_file_path):
        print("Downloading weights.zip...")
        response = requests.get(weights_url)

        with open(weights_zip_file_path, "wb") as f:
            f.write(response.content)

        print("Unzipping weights.zip...")
        with ZipFile(BytesIO(response.content), 'r') as zip_ref:
            zip_ref.extractall(clipseg_models_directory)

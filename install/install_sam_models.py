#install/install_sam_models.py
import os
import requests

def install():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    sam_models_directory = os.path.join(base_directory, "..", "..", "..", "models", "sams")

    if not os.path.exists(sam_models_directory):
        print(f"{sam_models_directory} does not exist. Creating.")
        os.makedirs(sam_models_directory)

    models_to_download = {
        "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    for model_filename, model_url in models_to_download.items():
        model_filepath = os.path.join(sam_models_directory, model_filename)
        if not os.path.exists(model_filepath):
            print(f"Downloading {model_filename}...")
            response = requests.get(model_url)
            with open(model_filepath, "wb") as f:
                f.write(response.content)

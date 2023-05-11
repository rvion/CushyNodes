# Api.py
import base64
import server
import os
import requests
from aiohttp import MultipartReader, web
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from urllib.parse import urlparse, unquote

cushy_nodes_path = os.path.abspath(__file__)
base_directory = os.path.dirname(cushy_nodes_path)
comfy_path = os.path.join(base_directory, "..", "..")
input_path = os.path.join(comfy_path, "input")
output_path = os.path.join(comfy_path, "output")
sam_models_directory = os.path.join(comfy_path, "models", "sams")

checkpoint = os.path.join(sam_models_directory, "sam_vit_h_4b8939.pth")
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)

def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_data_url(string):
    return string.startswith("data:")

def load_data_url(data_url):
    header, data = data_url.split(",", 1)
    if ";base64" in header:
        return base64.b64decode(data)
    else:
        return data.encode("utf-8")

@server.PromptServer.instance.routes.get("/cushy/sam/embeddings")
async def process_image(request):
    image_src = request.rel_url.query.get("image", None)

    if image_src is not None:
        image_src = unquote(image_src)
        print(image_src)

        if is_url(image_src):
            response = requests.get(image_src)
            image_data = response.content
        elif is_data_url(image_src):
            image_data = load_data_url(image_src)
        else:
            if not os.path.isabs(image_src):
                if os.path.dirname(image_src) == "":
                    image_src = os.path.join(input_path, image_src)
                else:
                    image_src = os.path.join(comfy_path, image_src)

            with open(image_src, "rb") as f:
                image_data = f.read()

        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(output_path, '__test.png'), image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()

        # Encode the embedding as base64
        embedding_data = image_embedding.tobytes()
        embedding_base64 = base64.b64encode(embedding_data).decode("utf-8")

        # You can save the embedding to a file or send it in the response
        #np.save("image_embedding.npy", image_embedding)

        return web.Response(status=200, text=embedding_base64)

    else:
        print("No image received.")
        return web.Response(status=400, text="No image received.")

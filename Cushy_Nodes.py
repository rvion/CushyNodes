# Cushy_Nodes.py
import torch
import cv2
import requests
import os
import clip
import numpy as np
from io import BytesIO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from models.clipseg.clipseg import CLIPDensePredT
from torchvision import transforms
from PIL import Image


# Set up paths
cushy_nodes_path = os.path.abspath(__file__)
base_directory = os.path.dirname(cushy_nodes_path)
sam_models_directory = os.path.join(base_directory, "..", "..", "models", "sams")
clipseg_models_directory = os.path.join(base_directory, "..", "..", "models", "clipseg", "clipseg_weights")

def get_files_in_directory(directory):
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    return files

# Image ------------------------------
class Cushy_Load_Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_url": ("STRING", {
                    "multiline": False,
                    "default": "https://images.pexels.com/photos/982300/pexels-photo-982300.jpeg"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "load_image"

    CATEGORY = "CushyNodes/Image"

    def load_image(self, image_url):
        response = requests.get(image_url)
        input_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)

        if input_image is None:
            raise ValueError("Unable to load image. Please check the image URL.")
        
        if input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGBA)
        else:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGRA2RGBA)

        input_image_expanded = np.expand_dims(input_image, axis=0)
        input_image_expanded = input_image_expanded.astype(np.float32)
        input_image_expanded /= 255.0
        image_tensor = torch.from_numpy(input_image_expanded)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        return (image_tensor,)

class Cushy_Resize_Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "resize_image"

    CATEGORY = "CushyNodes/Image"

    def resize_image(self, image, width, height):
        image = image.squeeze(0).cpu().numpy()
        image_uint8 = (image * 255.0).astype(np.uint8)
        resized_image = cv2.resize(image_uint8, (width, height), interpolation=cv2.INTER_AREA)
        resized_image_expanded = np.expand_dims(resized_image, axis=0)
        resized_image_expanded = resized_image_expanded.astype(np.float32)
        resized_image_expanded /= 255.0
        image_tensor = torch.from_numpy(resized_image_expanded)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        return (image_tensor,)

class Cushy_Resize_Image_By_Factor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", {"default": 0.5}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "resize_image_by_factor"

    CATEGORY = "CushyNodes/Image"

    def resize_image_by_factor(self, image, factor):
        image = image.squeeze(0).cpu().numpy()
        image_uint8 = (image * 255.0).astype(np.uint8)
        new_width = int(image_uint8.shape[1] * factor)
        new_height = int(image_uint8.shape[0] * factor)
        resized_image = cv2.resize(image_uint8, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_image_expanded = np.expand_dims(resized_image, axis=0)
        resized_image_expanded = resized_image_expanded.astype(np.float32)
        resized_image_expanded /= 255.0
        image_tensor = torch.from_numpy(resized_image_expanded)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        return (image_tensor,)

class Cushy_Select_Image_Index:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "index": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "select_image"

    CATEGORY = "CushyNodes/Image"

    def select_image(self, images, index):
        if index < 0 or index >= images.shape[0]:
            raise ValueError(f"Invalid index. Index should be between 0 and the number of images ({images.shape[0]}).")
        selected_image = images[index]
        selected_image = selected_image.unsqueeze(0)
        #selected_image = np.expand_dims(selected_image, axis=0)
        return (selected_image,)

# /Image ------------------------------

# Mask ------------------------------

class Cushy_Select_Mask_Index:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASKS",),
                "index": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("mask", "index",)

    FUNCTION = "select_mask"

    CATEGORY = "CushyNodes/Mask"

    def select_mask(self, masks, index):
        if index < 0 or index >= masks.shape[0]:
            raise ValueError(f"Invalid index. Index should be between 0 and the number of masks ({masks.shape[0]}).")
        selected_mask = masks[index]
        return (selected_mask, index)


class Cushy_Select_Mask_CLIP:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masked_images": ("IMAGE",),
                "text": ("STRING", {"default": "cat"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("mask", "index",)

    FUNCTION = "clip_select_mask"

    CATEGORY = "CushyNodes/Mask"

    def clip_select_mask(self, masked_images, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the CLIP model
        clip_model, preprocess = clip.load('ViT-B/32')
        clip_model = clip_model.to(device)

        # Preprocess the input text
        input_text = clip.tokenize(text)
        input_text = input_text.to(device)

        # Calculate text features
        with torch.no_grad():
            text_features = clip_model.encode_text(input_text)
            text_features = text_features.to(device)

        # Initialize the best match
        best_match_index = -1
        best_match_similarity = -float("inf")

        # Loop through the masked images and compare them to the input text
        for i, masked_image in enumerate(masked_images):
            # Preprocess the masked image
            image = Image.fromarray((masked_image.cpu().numpy() * 255).astype(np.uint8))
            preprocessed_image = preprocess(image).unsqueeze(0)
            preprocessed_image = preprocessed_image.to(device)  # Add this line

            # Calculate image features
            with torch.no_grad():
                image_features = clip_model.encode_image(preprocessed_image)

            # Calculate the similarity between the text and image features
            similarity = torch.nn.functional.cosine_similarity(text_features, image_features)

            # Update the best match if the current similarity is higher
            if similarity > best_match_similarity:
                best_match_similarity = similarity
                best_match_index = i

        # Return the best matching mask
        best_mask = masked_images[best_match_index]
        best_mask = best_mask[None,:,:,:]
        return (best_mask, best_match_index)

class Cushy_Select_Mask_Overlap:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASKS",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("mask", "index",)

    FUNCTION = "find_largest_overlap_mask"

    CATEGORY = "CushyNodes/Mask"

    def find_largest_overlap_mask(self, masks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure input masks have the same spatial dimensions
        assert masks.shape[1:] == mask.shape, "Input masks should have the same spatial dimensions"

        # Move tensors to the specified device
        masks = masks.to(device)
        mask = mask.to(device)

        # Calculate the intersection (overlap)
        intersection = (masks * mask).sum(dim=(1, 2))

        # Calculate the union (overlap + non-overlapping parts)
        union = masks.sum(dim=(1, 2)) + mask.sum() - intersection

        # Calculate the IoU (Intersection over Union)
        iou_scores = intersection.float() / union.float()

        # Find the mask with the largest IoU score
        best_match_idx = iou_scores.argmax().item()
        best_match_mask = masks[best_match_idx]

        return (best_match_mask, best_match_idx)




# /Mask ------------------------------


# AI ------------------------------

class Cushy_SAM_Segment_All:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "vit_model": (["default", "vit_h", "vit_l", "vit_b"],),
                "sam_model_name": (get_files_in_directory(sam_models_directory), { "default": "sam_vit_h_4b8939.pth" }),
            },
        }

    RETURN_TYPES = ("MASKS", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("masks", "mask_images", "images")

    FUNCTION = "execute_segmentation"

    CATEGORY = "CushyNodes"

    def execute_segmentation(self, image, vit_model, sam_model_name):
        # Load the SAM model
        sam_checkpoint = os.path.join(sam_models_directory, sam_model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[vit_model](checkpoint=sam_checkpoint)
        sam.to(device=device)

        # Use the input image
        input_image = image.squeeze(0).cpu().numpy()
        input_image_uint8 = (input_image * 255.0).astype(np.uint8)
        input_image_rgb = cv2.cvtColor(input_image_uint8, cv2.COLOR_RGBA2RGB)

        # Generate masks
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(input_image_rgb)

        # Convert masks to tensor
        masks_tensor = torch.stack([torch.from_numpy(mask['segmentation']).float() for mask in masks], dim=0)

        # Create tensor for black and white mask images
        mask_images_bw = masks_tensor[:, None, :, :].repeat(1, 3, 1, 1)
        mask_images_tensor = mask_images_bw.permute(0, 2, 3, 1)

        # Create tensor for masked images in color
        input_image_expanded = np.expand_dims(input_image, axis=0)
        input_image_tensor = torch.from_numpy(input_image_expanded).permute(0, 3, 1, 2).float()
        mask_images_color = input_image_tensor[:, :3, :, :] * masks_tensor[:, None, :, :]
        images_tensor = mask_images_color.permute(0, 2, 3, 1)

        return (masks_tensor, mask_images_tensor, images_tensor)


        
class Cushy_CLIP_Segmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "clip": ("STRING", {"default": "cat"}),
                "device": (sorted(['auto', 'cpu', 'cuda', 'mps', 'xpu']), { "default": 'auto' } ),
                "max_side": ("INT", {"default": 352, "min": 0, "max": 2048, "step": 8}),
                "threshold": ("INT", {"default": -1, "min": -1, "max": 255, "step": 1} ),
                "mode": (sorted(['average', 'sum']), { "default": 'sum' } ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "evaluate"

    CATEGORY = "CushyNodes"

    def evaluate(self, image, clip, device, max_side, threshold, mode):
        device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device)

        # Compute the new height and width based on max_side
        original_height, original_width = image.shape[1], image.shape[2]
        scaling_factor = max_side / max(original_height, original_width)
        new_height, new_width = int(original_height * scaling_factor), int(original_width * scaling_factor)

        # Load the model and weights
        model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
        model = model.to(device)
        model.eval()
        model.load_state_dict(torch.load(os.path.join(clipseg_models_directory, 'rd64-uni-refined.pth'), map_location=torch.device(device)), strict=False)

        # Transform input image
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((max_side, max_side), antialias=True),
        ])
        transformed_image = transform(image[..., :3].permute(0, 3, 1, 2))  # Change shape to [N, C, H, W] and remove alpha channel
        transformed_image = transformed_image.to(device)

        # Get predictions for each prompt
        prompts = clip.split(',')
        height, width = transformed_image.shape[2], transformed_image.shape[3]
        summed = torch.zeros([1, max_side, max_side], device=device)
        for prompt in prompts:
            pred = torch.zeros([1, max_side, max_side], device=device)
            with torch.no_grad():
                pred = model(transformed_image, prompt)[0]
            summed += torch.sigmoid(pred).squeeze(0)

        # Calculate averaged and summed masks
        averaged = (summed / len(prompts)).clamp(min=0, max=1).squeeze(0)
        summed = summed.clamp(min=0, max=1).squeeze(0)

        # Apply threshold if specified
        if threshold >= 0:
            averaged = (averaged - threshold / 255).clamp(min=0, max=1).ceil()
            summed = (summed - threshold / 255).clamp(min=0, max=1).ceil()

        # Create output image and mask
        mask = averaged if mode == "average" else summed
        image_out = torch.stack([mask] * 3).permute(1, 2, 0).unsqueeze(0)  # Shape: [N, H, W, C]
        
        # Resize mask and image_out back to the original size
        resize_back = transforms.Resize((original_height, original_width), antialias=True)
        mask = resize_back(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        image_out = resize_back(image_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


        return (image_out, mask,)
# /AI ------------------------------








# Add the new node to the NODE_CLASS_MAPPINGS dictionary
NODE_CLASS_MAPPINGS = {
    #AI
    "SAM Segment All": Cushy_SAM_Segment_All,
    "CLIP Segmentation": Cushy_CLIP_Segmentation,
    #Masks
    "Select Mask Index": Cushy_Select_Mask_Index,
    "Select Mask CLIP": Cushy_Select_Mask_CLIP,
    "Select Mask Overlap": Cushy_Select_Mask_Overlap,
    #Image
    "Load Image": Cushy_Load_Image,
    "Resize Image": Cushy_Resize_Image,
    "Resize Image By Factor": Cushy_Resize_Image_By_Factor,
    "Select Image Index": Cushy_Select_Image_Index,
}

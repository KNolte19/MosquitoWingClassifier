import numpy as np
import tensorflow as tf
import skimage as ski
from rembg import remove, new_session
from PIL import Image
import torch
import torchvision

def process_image(file_stream, processed_img_path, bg_session):
    """Process an image by removing the background, aligning, enhancing contrast, and resizing."""
        
    def remove_bg_and_rotate(image, bg_session):
        # Remove the background (transparent background)
        image_rembg = remove(image, bgcolor=(0, 0, 0, -1), session=bg_session)
        mask = np.asarray(image_rembg)[:, :, 3] > 10  # Mask where alpha > 10

        # Apply the mask to the image to retain only foreground
        image_masked = image_rembg * np.dstack([mask] * 4)

        # Get orientation from region properties
        properties = ski.measure.regionprops_table(ski.measure.label(mask), properties=("axis_major_length", "orientation"))
        angle = -(properties["orientation"][np.argmax(properties["axis_major_length"])] * (180 / np.pi) + 90)
        angle = angle + 180 if angle < -90 else angle  # Normalize the angle to a valid range

        # Rotate the image and mask based on the calculated angle
        rotated_image = ski.transform.rotate(image_masked, angle, resize=False, mode='edge', preserve_range=True)
        rotated_mask = ski.transform.rotate(mask, angle, resize=False, mode='edge', preserve_range=True)

        # Remove empty rows and columns (crop the image to the non-empty region)
        rows = np.any(rotated_mask, axis=1)
        cols = np.any(rotated_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the image and mask to the bounding box of non-empty regions
        image_cropped = rotated_image[rmin:rmax, cmin:cmax]
        mask_cropped = rotated_mask[rmin:rmax, cmin:cmax]

        # Return the cropped image and mask (exclude alpha channel from the image)
        return image_cropped[:, :, :3], mask_cropped
    
    def CLAHE_transform(image, clip_limit=0.5, nbins=32):
        # Convert image to grayscale and apply CLAHE
        equalized_img = ski.exposure.equalize_adapthist(np.mean(image, axis=-1), clip_limit=clip_limit, nbins=nbins)
        equalized_img = ski.filters.median(equalized_img, ski.morphology.disk(1))
        return torch.tensor(equalized_img, dtype=torch.float32)


    def pad_and_resize(image):
        # Pad the image to make it square, with the longer dimension being the target size
        height, width = image.shape[:2]
        max_dim = max(height, width)
        pad_height = (max_dim - height) // 2
        pad_width = (max_dim - width) // 2
        image_padded = torch.nn.functional.pad(image, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)

        # Resize the image to 512x512, then crop to the desired region
        image_resized = torchvision.transforms.functional.resize(image_padded.unsqueeze(0), (512, 512)).numpy()[0]
        return image_resized[128:384, :]

    # Load image
    image = Image.open(file_stream)

    # Remove background, rotate the image, and get the mask
    image, mask = remove_bg_and_rotate(image, bg_session)

    # Apply CLAHE transformation
    image = CLAHE_transform(image / 255)

    # Apply the mask to remove background from image
    image[~mask] = 0

    # Pad and resize the image to the specified size
    image = pad_and_resize(image)

    # Save the processed image
    ski.io.imsave(processed_img_path, (image * 255).astype(np.uint8))

    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

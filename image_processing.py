import numpy as np
import skimage as ski
from rembg import remove
from PIL import Image
import torch
import torchvision
import albumentations as A
import warnings
warnings.filterwarnings("ignore")

class ImageGenerator (torch.utils.data.Dataset):
    def __init__(self, file_list, N_augmentations, processed_file_name_list, bg_session):
        self.file_list = file_list
        self.N_augmentations = N_augmentations
        self.processed_file_name_list = processed_file_name_list
        self.bg_session = bg_session

        self.augment_pipe = A.Compose([
                            # Image Capture Variance
                            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=.5), # Had to be removed due to dependency issues
                            A.PlanckianJitter(p=.5), # Had to be removed due to dependency issues
                            A.ImageCompression(quality_lower=75, quality_upper=100, p=.25),
                            A.Defocus(radius=(1, 3), p=.25),
                            A.RandomGamma(gamma_limit=(80, 120), p=.25),
                            A.MotionBlur(blur_limit=(3, 3), p=.25),
                            A.Downscale(scale_min=0.75, scale_max=.99, p=.25),
                            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=.5),
                            A.ChannelDropout(channel_drop_range=(1, 1), p=.25),
                            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=.25),
                        ])

    def __len__(self):
        return len(self.file_list)
    
    def remove_bg_and_rotate(self, image, bg_session):
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
    
    def CLAHE_transform(self, image, clip_limit=0.5, nbins=32):
        # Convert image to grayscale and apply CLAHE
        equalized_img = ski.exposure.equalize_adapthist(np.mean(image, axis=-1), clip_limit=clip_limit, nbins=nbins)
        equalized_img = ski.filters.median(equalized_img, ski.morphology.disk(1))
        return torch.tensor(equalized_img, dtype=torch.float32)


    def pad_and_resize(self, image):
        # Pad the image to make it square, with the longer dimension being the target size
        height, width = image.shape[:2]
        max_dim = max(height, width)
        pad_height = (max_dim - height) // 2
        pad_width = (max_dim - width) // 2
        image_padded = torch.nn.functional.pad(image, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)

        # Resize the image to 512x512, then crop to the desired region
        image_resized = torchvision.transforms.functional.resize(image_padded.unsqueeze(0), (384, 384)).numpy()[0]
        return image_resized[96:288, :]

    def __getitem__(self, idx):
        """
        Generate an image tensor based on the index.
        Args:
            idx: Index of the augmentation (0 for the original image).

        Returns:
            Tensor of the processed image.
        """

        augmented_datasets = []
        # Load image
        file = self.file_list[idx].stream
        image = Image.open(file)

        # Remove background, rotate the image, and get the mask
        image, mask = self.remove_bg_and_rotate(image, self.bg_session)

        # Standardize the image to have pixel values in the range [0, 1]
        image = np.asarray(image).astype('float32') / 255

        for i in range(self.N_augmentations):
            if i != 0:
                # Apply augmentations to the image
                processed_image = self.augment_pipe(image = image)["image"]
            else:
                # Use the original unaugmented image
                processed_image = image

            # Normalize and apply CLAHE
            processed_image = self.CLAHE_transform(processed_image)

            # Apply mask to remove the background
            processed_image[~mask] = 0

            # Pad and resize the image
            processed_image = self.pad_and_resize(processed_image)

            # Convert to tensor and add batch dimension
            processed_tensor = torch.tensor(processed_image, dtype=torch.float32).unsqueeze(0)

            augmented_datasets.append(processed_tensor)

            if i == 0:
                ski.io.imsave(self.processed_file_name_list[idx], (processed_image * 255).astype(np.uint8))

        return augmented_datasets
import numpy as np
import tensorflow as tf
import skimage as ski
from rembg import remove, new_session
from PIL import Image
import keras_cv

# Initialize a new session for background removal
bg_session = new_session()

augmentation_model = tf.keras.Sequential([tf.keras.layers.RandomBrightness((-.8, .1)),
                                          tf.keras.layers.GaussianNoise(.25),
                                          tf.keras.layers.GaussianDropout(.25),
                                          keras_cv.layers.RandomSaturation((0,1)),
                                          keras_cv.layers.RandomHue(.5, value_range=(0, 255)),
                                          keras_cv.layers.RandomColorDegeneration(.5),
                                          keras_cv.layers.RandomSharpness(.5, value_range=(0, 255)),])


def process_image(file_stream, bg_session):
    """Process an image by removing the background, aligning, enhancing contrast, and resizing."""

    def remove_bg_and_rotate(image):
        """
        Remove the background from the image and rotate it to align with the main axis.

        Args:
            image (PIL.Image): Input image to process.

        Returns:
            tuple: A tuple containing the mask and the cropped, aligned image.
        """
        # Remove Background
        image_rembg = remove(image, bgcolor=(0, 0, 0, -1), session=bg_session)
        mask = np.asarray(image_rembg)[:, :, 3] > 10

        # Get Angle through Regionprops
        image_masked = image_rembg * np.dstack([mask] * 4)
        properties = ski.measure.regionprops_table(
            ski.measure.label(mask), properties=("axis_major_length", "orientation")
        )
        angle = -(
            properties["orientation"][np.argmax(properties["axis_major_length"])]
            * (180 / np.pi)
            + 90
        )
        if angle < -90:
            angle += 180

        # Rotate image and mask
        rotated_image = np.asarray(Image.fromarray(image_masked).rotate(angle))
        rotated_mask = np.asarray(Image.fromarray(mask).rotate(angle))

        # Remove empty pixel rows and columns
        rows = np.any(rotated_mask, axis=1)
        cols = np.any(rotated_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the image to the bounding box
        image_cropped = rotated_image[rmin:rmax, cmin:cmax]
        image_cropped = image_cropped[:, :, :3]  # Keep only RGB channels
        mask_cropped = rotated_mask[rmin:rmax, cmin:cmax]

        return mask_cropped, image_cropped

    def process_image_to_grey(image):
        """
        Convert the image to greyscale.

        Args:
            image (numpy.ndarray): Input RGB image.

        Returns:
            numpy.ndarray: Greyscale image.
        """
        image = ski.color.rgb2gray(image)
        return ski.util.img_as_ubyte(np.asarray(image))

    def enhance_contrast_and_resize(mask, image):
        """
        Enhance contrast using CLAHE and resize the image.

        Args:
            mask (numpy.ndarray): Binary mask of the image.
            image (numpy.ndarray): Greyscale image.

        Returns:
            numpy.ndarray: Enhanced and resized image.
        """
        # Reduce noise
        image = ski.filters.rank.median(image, ski.morphology.disk(3))

        # Apply CLAHE
        equalized_img = np.asarray(
            ski.exposure.equalize_adapthist(image, clip_limit=0.6, nbins=48)
        )
        equalized_img = ski.filters.rank.median(
            ski.util.img_as_ubyte(equalized_img), ski.morphology.disk(3)
        )

        # Set background to 0
        equalized_img[~mask] = 0

        # Resize image and crop it to size
        resized_img = tf.image.resize_with_pad(
            np.stack((equalized_img,) * 3, axis=-1), 300, 300
        ).numpy()#[:, :, 0]

        return resized_img

    # Open image
    image_ls = []
    image = Image.open(file_stream)

    # Remove background and align wing
    mask, image = remove_bg_and_rotate(image)

    for i in range(4):
        # Dont augment the first image
        if i == 0:
            image_aug = np.asarray(image)
        else:
            image_aug = augmentation_model(np.asarray(image)).numpy()

        # Transform image to greyscale
        grey_image = process_image_to_grey(image_aug / 255)

        # Enhance contrast and resize image
        clahe_image = enhance_contrast_and_resize(mask, grey_image)

        #Flip every second image
        if i%2 == 0:
            clahe_image = np.fliplr(clahe_image)

        unaugment_image = Image.fromarray(np.uint8(clahe_image))
        
        image_ls.append(clahe_image)

    return image_ls, unaugment_image

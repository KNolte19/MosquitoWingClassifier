import pandas as pd
import numpy as np
import tensorflow as tf
import os 
import keras_cv

from skimage.measure import label, regionprops_table
from rembg import remove, new_session
from PIL import Image

def process_image(src_img_path, session):
    def remove_bg_and_rotate(image):
        # Remove Background
        image_rembg = remove(image, bgcolor=(0, 0, 0, -1), session=session)
        mask = np.asarray(image_rembg)[:,:,3] > 10

        # Get Angle through Regionprops  
        image_masked = image_rembg * np.dstack([mask]*4)
        properties = regionprops_table(label(mask), properties=("axis_major_length", "orientation"))
        angle = -(properties["orientation"][np.argmax(properties["axis_major_length"])] * (180/np.pi) + 90)
        if angle < -90: 
            angle = angle + 180

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
        image_cropped = image_cropped[:,:,:3]
        
        return image_cropped

    def add_blank_background(image):
        height, width, channels = image.shape
        max_size = np.max([height, width])
    
        white_background = np.full((max_size, max_size, channels), 0, dtype=np.uint8)
        paste_position = (0, int((max_size-height)/2))
        white_background[paste_position[1]:paste_position[1]+height, paste_position[0]:paste_position[0]+width] = image
        padded_image = tf.image.resize_with_pad(white_background, 2000, 2000).numpy()
        padded_image_uint = padded_image.astype(np.uint8)
        padded_image_img = Image.fromarray(padded_image_uint)
        return padded_image_img
        
    input = Image.open(src_img_path)
    rotated_image = remove_bg_and_rotate(input)
    image_filled = add_blank_background(rotated_image)

    return image_filled
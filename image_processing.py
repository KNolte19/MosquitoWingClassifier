import numpy as np
import tensorflow as tf
import skimage as ski 
from rembg import remove, new_session
from PIL import Image
bg_session = new_session()

def process_image(file_path, bg_session):
    def remove_bg_and_rotate(image):
        # Remove Background
        image_rembg = remove(image, bgcolor=(0, 0, 0, -1), session=bg_session)
        mask = np.asarray(image_rembg)[:,:,3] > 10

        # Get Angle through Regionprops  
        image_masked = image_rembg * np.dstack([mask]*4)
        properties = ski.measure.regionprops_table(ski.measure.label(mask), properties=("axis_major_length", "orientation"))
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
        mask_cropped = rotated_mask[rmin:rmax, cmin:cmax]
        
        return mask_cropped, image_cropped

    def process_image_to_grey(image):
        # make greyscale
        image = ski.color.rgb2gray(image)
        image = ski.util.img_as_ubyte(np.asarray(image))
        return image

    def process_image(mask, image):
        # reduce noise
        image = ski.filters.rank.median(image, ski.morphology.disk(3))
        # apply CLAHE
        equalized_img = np.asarray(ski.exposure.equalize_adapthist(image,
                                                                   clip_limit=.6,
                                                                   nbins=48))
        
        equalized_img = ski.filters.rank.median(ski.util.img_as_ubyte(equalized_img), ski.morphology.disk(3))
        
        # set background to 0
        equalized_img[~mask] = 0
        # resize image and crop it to size
        resized_img = tf.image.resize_with_pad(np.stack((equalized_img,)*3, axis=-1), 300, 300, ).numpy()[:,:,0]
        return resized_img

    # open image
    image = Image.open(file_path)

    # remove background and align wing
    mask, image = remove_bg_and_rotate(image)
            
    # transform image to greyscale
    image = process_image_to_grey(image/255)
    
    # apply CLAHE for contrast improvement 
    image = process_image(mask, image)
    image = Image.fromarray(np.uint8(image))
    
    return image 
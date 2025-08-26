#!/usr/bin/env python3
"""
Color Normalization Module

This module is replicating the color normalization method used in the HistomicsTK package (https://github.com/DigitalSlideArchive/HistomicsTK).
"""

import numpy as np
import cv2
from typing import Optional

# define conversion matrices
_rgb2lms = np.array([[0.3811, 0.5783, 0.0402],
                     [0.1967, 0.7244, 0.0782],
                     [0.0241, 0.1288, 0.8444]])

_lms2lab = np.dot(
    np.array([[1 / (3**0.5), 0, 0],
              [0, 1 / (6**0.5), 0],
              [0, 0, 1 / (2**0.5)]]),
    np.array([[1, 1, 1],
              [1, 1, -2],
              [1, -1, 0]]),
)

# Define conversion matrices
_lms2rgb = np.linalg.inv(_rgb2lms)
_lab2lms = np.linalg.inv(_lms2lab)



def rgb_to_lab(im_rgb):
    """Transforms an image from RGB to LAB color space

    Parameters
    ----------
    im_rgb
        An RGB image

    Returns
    -------
    im_lab
        LAB representation of the input image `im_rgb`.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.lab_to_rgb,
    histomicstk.preprocessing.color_normalization.reinhard

    References
    ----------
    .. [#] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone
       responses to natural images: implications for visual coding,"
       J. Opt. Soc. Am. A vol.15, pp.2036-2045, 1998.

    """
    # get input image dimensions
    m = im_rgb.shape[0]
    n = im_rgb.shape[1]

    # calculate im_lms values from RGB
    im_rgb = np.reshape(im_rgb, (m * n, 3))
    im_lms = np.dot(_rgb2lms, np.transpose(im_rgb))
    im_lms[im_lms == 0] = np.spacing(1)

    # calculate LAB values from im_lms
    im_lab = np.dot(_lms2lab, np.log(im_lms))

    # reshape to 3-channel image
    im_lab = np.reshape(im_lab.transpose(), (m, n, 3))

    return im_lab

def lab_to_rgb(im_lab):
    """Transforms an image from LAB to RGB color space

    Parameters
    ----------
    im_lab
        An image in LAB color space

    Returns
    -------
    im_rgb
        The RGB representation of the input image 'im_lab'.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.rgb_to_lab,
    histomicstk.preprocessing.color_normalization.reinhard

    References
    ----------
    .. [#] D. Ruderman, T. Cronin, and C. Chiao, "Statistics of cone
       responses to natural images: implications for visual coding,"
       J. Opt. Soc. Am. A 15, 2036-2045 (1998).

    """
    # get input image dimensions
    m = im_lab.shape[0]
    n = im_lab.shape[1]

    # calculate im_lms values from LAB
    im_lab = np.reshape(im_lab, (m * n, 3))
    im_lms = np.dot(_lab2lms, np.transpose(im_lab))

    # calculate RGB values from im_lms
    im_lms = np.exp(im_lms)
    im_lms[im_lms == np.spacing(1)] = 0

    im_rgb = np.dot(_lms2rgb, im_lms)

    # reshape to 3-channel image
    im_rgb = np.reshape(im_rgb.transpose(), (m, n, 3))

    return im_rgb

def reinhard(im_src: np.ndarray, 
             target_mu: np.ndarray, 
             target_sigma: np.ndarray,
             mask_out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Perform Reinhard color normalization on an RGB image.
    
    This method normalizes the color distribution of an image by matching
    the mean and standard deviation of its LAB color space representation
    to target values.

    Parameters
    ----------
    im_src
        Source RGB image with values in range [0, 1] and shape (H, W, 3)
    target_mu
        Target mean values for L, A, B channels (shape: 3)
    target_sigma
        Target standard deviation values for L, A, B channels (shape: 3)
    mask_out
        Boolean mask indicating pixels to exclude from statistics calculation.
        True values are excluded. Shape should match (H, W).
        
    Returns
    -------
    np.ndarray
        Color normalized RGB image with values in range [0, 1]
    """
    # Convert source image to LAB
    im_lab = rgb_to_lab(im_src)

    # mask out irrelevant tissue / whitespace / etc
    if mask_out is not None:
        mask_out = mask_out[..., None]
        im_lab = np.ma.masked_array(
            im_lab, mask=np.tile(mask_out, (1, 1, 3)))

    src_mu = [im_lab[..., i].mean() for i in range(3)]
    src_sigma = [im_lab[..., i].std() for i in range(3)]

    # scale to unit variance
    for i in range(3):
        im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i]

    # rescale and recenter to match target statistics
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] * target_sigma[i] + target_mu[i]

    # convert back to RGB colorspace
    im_normalized = lab_to_rgb(im_lab)
    im_normalized[im_normalized > 255] = 255
    im_normalized[im_normalized < 0] = 0

    # return masked values and reconstruct unmasked LAB image
    if mask_out is not None:
        im_normalized = im_normalized.data
        for i in range(3):
            original = im_src[:, :, i].copy()
            new = im_normalized[:, :, i].copy()
            original[np.not_equal(mask_out[:, :, 0], True)] = 0
            new[mask_out[:, :, 0]] = 0
            im_normalized[:, :, i] = new + original
    im_normalized = im_normalized.astype(np.uint8)

    return im_normalized

def color_normalize(image: np.ndarray, 
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply Reinhard color normalization with predefined H&E staining parameters.
    
    Parameters
    ----------
    image
        Input RGB image with values in range [0, 1] and shape (H, W, 3)
    mask
        Boolean mask where True indicates tissue regions. 
        Normalization statistics are computed only from tissue regions.
        
    Returns
    -------
    np.ndarray
        Color normalized RGB image with values in range [0, 1]
    """
    # Predefined H&E staining target parameters
    cnorm = {
        'mu': np.array([8.74108109, -0.12440419, 0.0444982]),
        'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
    }
    
    # Create mask_out (pixels to exclude from statistics)
    # mask==0 means background pixels should be excluded
    mask_out = None if mask is None else (mask == 0)
    
    # Apply Reinhard normalization
    normalized_image = reinhard(
        image,
        target_mu=cnorm['mu'], 
        target_sigma=cnorm['sigma'],
        mask_out=mask_out
    )

    normalized_image = (normalized_image / np.max(normalized_image, axis=(0, 1), keepdims=True)).astype(np.float32)
    return normalized_image

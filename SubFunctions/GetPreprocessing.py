import cv2
import numpy as np
from dataclasses import dataclass



@dataclass
class Preprocessing(object):
    # Initialize with the input image
    def __init__(self, image):
        self.image = image

    # Static method for preprocessing the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    @staticmethod
    def preprocessing(image):
        # Print status message for preprocessing

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge channels
        limg = cv2.merge((cl, a, b))

        # Convert back to BGR
        clahe_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # Return the CLAHE processed image
        return clahe_image

    # Static method for extracting the Region of Interest (ROI) from the image
    @staticmethod
    def roi_extraction(image):
        # Print status message for ROI extraction

        # Convert the image to grayscale for easier thresholding and contour detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Optional: blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Threshold
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or ADAPTIVE_THRESH_MEAN_C
            cv2.THRESH_BINARY_INV,
            11,  # block size (must be odd)
            2  # constant subtracted from mean
        )

        # Optional: Morphological cleaning
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(max_contour)

            # Extract ROI
            roi = image[y:y + h, x:x + w]
            return roi
        else:
            return image



    @staticmethod
    def hair_removal(image):

        # Convert the original image to grayscale
        grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Kernel for the morphological filtering
        kernel = cv2.getStructuringElement(1, (17, 17))

        # Perform the blackHat filtering on the grayscale image to find the
        # hair countours
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

        # intensify the hair countours in preparation for the inpainting
        # algorithm
        ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # inpaint the original image depending on the mask
        dst = cv2.inpaint(image, thresh2, 1, cv2.INPAINT_TELEA)

        return dst

    # Method to apply preprocessing and ROI extraction sequentially
    def get_preprocessing(self):
        # Apply preprocessing to the image
        hair_removed = self.hair_removal(self.image)
        roi_image = self.roi_extraction(hair_removed)
        preprocessed = self.preprocessing(roi_image)
        preprocessed = cv2.resize(preprocessed, (96, 96))
        return preprocessed

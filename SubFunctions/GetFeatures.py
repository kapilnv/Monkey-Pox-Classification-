import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from keras.models import Model
from termcolor import cprint
from keras.applications.densenet import DenseNet169
from keras.applications.efficientnet import EfficientNetB7

from scipy.stats import skew, kurtosis, entropy
from scipy.spatial import distance

from SubFunctions.LBP import LBP
from SubFunctions.LGP import LGP
from SubFunctions.LDP import LDP
from keras.layers import *

from SubFunctions.StructuralPattern import StructuralPattern
from SubFunctions.Attention import CrossAttention

densenet169 = DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
efficientnetb7 = EfficientNetB7(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

x1 = densenet169.layers[4].output  # (None, 112, 112, 64)
x2 = efficientnetb7.layers[5].output  # (None, 112, 112, 64)
x1 = Reshape(target_shape=(112, 112*64))(x1)
x2 = Reshape(target_shape=(112, 112*64))(x2)
x = CrossAttention()([x1, x2])
x = Reshape(target_shape=(112, 112, 64))(x)
custom_model = Model(inputs=[densenet169.input, efficientnetb7.input], outputs=x)




@dataclass
class FeatureExtraction(object):
    def __init__(self, image: np.ndarray):
        self.image = image



    @staticmethod
    def distance_based_fusing(image1, image2):
        height, width = image1.shape
        img_hybrid = np.zeros((height, width), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                dis1 = distance.euclidean([image1[i, j]], [image2[i, j]])
                dis2 = distance.jaccard([image1[i, j]], [image2[i, j]])
                img_hybrid[i, j] = np.mean([dis1, dis2])

        return img_hybrid







    def distance_features(self, image) -> np.ndarray:
        cprint("[⚠️] Hybrid distance aware structural pattern   ", color='yellow', attrs=['bold'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = LBP(image).get_lbp()
        gradient = LGP(image).get_lgp()
        gradient = cv2.resize(gradient.astype(np.uint8), (image.shape[1], image.shape[0]))
        features = self.distance_based_fusing(binary, gradient)
        return features


    @staticmethod
    def entropy_based_fusing(image1, image2, image3):
        height, width = image1.shape
        img_hybrid = np.zeros((height, width), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                img_hybrid[i, j] = entropy([image1[i, j], image2[i, j], image3[i, j]])
        return np.nan_to_num(img_hybrid)


    def colormap_structural(self, image):
        cprint("[⚠️] 3D colormap-based hybrid structural pattern  ", color='yellow', attrs=['bold'])
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        image3 = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        image1 = np.mean(image1, axis=2)
        image2 = np.mean(image2, axis=2)
        image3 = np.mean(image3, axis=2)

        pattern1 = StructuralPattern(image1).get_structural_pattern()
        pattern2 = StructuralPattern(image2).get_structural_pattern()
        pattern3 = StructuralPattern(image3).get_structural_pattern()


        hybrid = self.entropy_based_fusing(pattern1, pattern2, pattern3)

        return hybrid




    @staticmethod
    def pretrained_features(image) -> np.ndarray:
        cprint("[⚠️] Mutual Information layer fused pretrained features  ", color='yellow', attrs=['bold'])
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        outputs = np.squeeze(custom_model.predict([image, image]))

        outputs = cv2.resize(np.mean(outputs, axis=2), (96, 96))
        return outputs





    def get_features(self):
        distance_features = self.distance_features(self.image)
        colormap_features = self.colormap_structural(self.image)
        pretrained_features = self.pretrained_features(self.image)


        features = np.concatenate((np.expand_dims(distance_features, axis=-1),
                                   np.expand_dims(colormap_features, axis=-1),
                                   np.expand_dims(pretrained_features, axis=-1)), axis=-1)
        return features

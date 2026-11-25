import dataclasses
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
from termcolor import cprint
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
from SubFunctions.GetPreprocessing import Preprocessing
from SubFunctions.GetFeatures import FeatureExtraction


@dataclasses.dataclass
class ReadDataset(object):
    def __init__(self, dataset: int, exec: bool = True):
        self.dataset = dataset
        self.exec = exec

    def read_data(self):
        if self.exec:
            cprint(f"[⁉️] Extracting Features and Labels from Dataset {self.dataset} ", color='grey',
                   on_color='on_white')

            Features = []
            Labels = []


            if self.dataset == 1:

                path = glob.glob(f"DATASET\\Dataset1\\***\\**\\*.jpg")
                random.shuffle(path)

                for filename in tqdm(path, desc="Extracting Features"):

                    image = cv2.imread(filename)
                    preprocessed = Preprocessing(image).get_preprocessing()
                    features = FeatureExtraction(preprocessed).get_features()
                    Features.append(features)
                    if filename.split("\\")[-2] == 'healthy':

                        Labels.append(0)
                    else:
                        Labels.append(1)

                print(np.unique(Labels, return_counts=True))





            else:
                path = glob.glob(f"DATASET\\Dataset2\\**\\*.jpg")
                random.shuffle(path)

                for filename in tqdm(path, desc="Extracting Features"):
                    image = cv2.imread(filename)
                    preprocessed = Preprocessing(image).get_preprocessing()
                    features = FeatureExtraction(preprocessed).get_features()
                    Features.append(features)

                    if filename.split("\\")[-2] == 'Normal':
                        Labels.append(0)
                    else:
                        Labels.append(1)

                print(np.unique(Labels, return_counts=True))


            # Convert all features and labels to NumPy arrays
            Features = np.array(Features)
            Labels = np.array(Labels).astype(int)

            # Pack features and labels into a dictionary
            data = {
                'Features': Features,
                'Labels': Labels
            }

            os.makedirs(f'Features\\Dataset{self.dataset}', exist_ok=True)
            # Save the extracted features and labels as a pickle file for future use
            with open(f'Features\\Dataset{self.dataset}\\Features.pkl', 'wb') as f:
                pickle.dump(data, f)

            # Print confirmation message that feature extraction is complete
            cprint(f"[✅] Feature Extraction Done !! ", color='grey', on_color='on_white')

        else:
            # If features were already extracted earlier, load them from the pickle file
            cprint(f"[⁉️] Loading the Extracted Features and Labels ", color='grey', on_color='on_white')

            # Load the saved dictionary (features + labels) from the pickle file
            with open(f'Features\\Dataset{self.dataset}\\Features.pkl', 'rb') as f:
                data = pickle.load(f)

            # Print confirmation message that loading is complete
            cprint(f"[✅] Feature Loading Done !! ", color='grey', on_color='on_white')

        # Return the dictionary containing Features and Labels
        return data




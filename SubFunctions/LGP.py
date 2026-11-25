import cv2
import numpy as np


class LGP:
    def __init__(self, image):
        self.image = image
        self.r = 1

    def get_lgp(self):
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        padded = np.pad(self.image, (self.r, self.r), 'constant')
        a1 = padded[:-2 * self.r, :-2 * self.r]
        b1 = padded[:-2 * self.r, self.r:-self.r]
        a2 = padded[:-2 * self.r, 2 * self.r:]
        b2 = padded[self.r:-self.r, 2 * self.r:]
        a3 = padded[2 * self.r:, 2 * self.r:]
        b3 = padded[2 * self.r:, self.r:-self.r]
        a4 = padded[2 * self.r:, :-2 * self.r]
        b4 = padded[self.r:-self.r, :-2 * self.r]
        codes = (a1 >= a3) + 2 * (a2 >= a4) + 4 * (b1 >= b3) + 8 * (b2 >= b4)
        codes = codes[self.r:-self.r, self.r:-self.r]
        return codes


import cv2
import numpy as np
from termcolor import cprint





class LDP:
    def __init__(self, image):
        self.image = image
        
    @staticmethod
    def assign_bit(image, x, y, c1, c2, d):  # assign bit according to degree and neighbouring pixel
        # a and b are 1 if increasing and 0 if decreasing
        if (d == 0):
            a = 0
            b = 0
            try:
                if image[c1][c2 + 1] >= image[c1][c2]:
                    a = 1
                if image[x][y + 1] >= image[x][y]:
                    b = 1
            except:
                pass
        if (d == 45):
            a = 0
            b = 0
            try:
                if image[c1 - 1][c2 + 1] >= image[c1][c2]:
                    a = 1
                if image[x - 1][y + 1] >= image[x][y]:
                    b = 1
            except:
                pass
        if (d == 90):
            a = 0
            b = 0
            try:
                if image[c1 - 1][c2] >= image[c1][c2]:
                    a = 1
                if image[x - 1][y] >= image[x][y]:
                    b = 1
            except:
                pass
        if (d == 135):
            a = 0
            b = 0
            try:
                if image[c1 - 1][c2 - 1] >= image[c1][c2]:
                    a = 1
                if image[x - 1][y - 1] >= image[x][y]:
                    b = 1
            except:
                pass
        if (a == b):  # if monotonically increasing or decreasing than 0
            return "0"
        else:  # if turning point
            return "1"
        return bit
    
    
    def local_der_val(self, image, x, y):  # calculating local derivative pattern value of a pixel
        thirtytwo_bit_binary = []
        centre = image[x][y]
        c1 = x
        c2 = y
        decimal_val = 0
        # starting from top left,assigning bit to pixels clockwise at 0 degree
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y - 1, c1, c2, 0))
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y, c1, c2, 0))
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y + 1, c1, c2, 0))
        thirtytwo_bit_binary.append(self.assign_bit(image, x, y + 1, c1, c2, 0))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y + 1, c1, c2, 0))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y, c1, c2, 0))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y - 1, c1, c2, 0))
        thirtytwo_bit_binary.append(self.assign_bit(image, x, y - 1, c1, c2, 0))

        # starting from top left,assigning bit to pixels clockwise at 45 degree
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y - 1, c1, c2, 45))
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y, c1, c2, 45))
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y + 1, c1, c2, 45))
        thirtytwo_bit_binary.append(self.assign_bit(image, x, y + 1, c1, c2, 45))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y + 1, c1, c2, 45))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y, c1, c2, 45))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y - 1, c1, c2, 45))
        thirtytwo_bit_binary.append(self.assign_bit(image, x, y - 1, c1, c2, 45))

        # starting from top left,assigning bit to pixels clockwise at 90 degree
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y - 1, c1, c2, 90))
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y, c1, c2, 90))
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y + 1, c1, c2, 90))
        thirtytwo_bit_binary.append(self.assign_bit(image, x, y + 1, c1, c2, 90))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y + 1, c1, c2, 90))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y, c1, c2, 90))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y - 1, c1, c2, 90))
        thirtytwo_bit_binary.append(self.assign_bit(image, x, y - 1, c1, c2, 90))

        # starting from top left,assigning bit to pixels clockwise at 135 degree
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y - 1, c1, c2, 135))
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y, c1, c2, 135))
        thirtytwo_bit_binary.append(self.assign_bit(image, x - 1, y + 1, c1, c2, 135))
        thirtytwo_bit_binary.append(self.assign_bit(image, x, y + 1, c1, c2, 135))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y + 1, c1, c2, 135))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y, c1, c2, 135))
        thirtytwo_bit_binary.append(self.assign_bit(image, x + 1, y - 1, c1, c2, 135))
        thirtytwo_bit_binary.append(self.assign_bit(image, x, y - 1, c1, c2, 135))

        str1 = ""
        l = str1.join(thirtytwo_bit_binary)  # 32 bit binary number
        decimal_val = int(l, 2)  # 32 bit binary to decimal number
        return decimal_val


    def get_ldp(self):

        if len(self.image.shape) != 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        m, n = self.image.shape
        # gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
        ldp_image = np.zeros((m, n))
        # ldp_image = np.zeros((m, n))
        # converting image to ldp
        for i in range(0, m):
            for j in range(0, n):
                ldp_image[i, j] = int(self.local_der_val(self.image, i, j))
    
        return ldp_image

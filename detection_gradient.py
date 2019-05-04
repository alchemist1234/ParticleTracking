import numpy as np  # Arrays
import cv2  # OpenCV


############ SPECIFIC FUNCTIONS ###############

class GradientClass(object):

    def __init__(self):
        pass

    def get_sobel(self, img, dirx, diry):
        # Return thresholded sobel in specified direction
        sobel = cv2.Sobel(img, cv2.CV_64F, dirx, diry, ksize=self.Vars['sobel_ksize'])
        sobel = np.absolute(sobel)
        sobel = sobel - sobel.min()  # Rescale to 0-255 values
        thresh = np.zeros(sobel.shape, np.uint8)
        thresh[sobel > self.Vars['sobel_thresh']] = 255
        sobel = thresh.copy()

        sobel = self.morphology(sobel)

        return sobel

    def binarization(self, frame_gray, *args):
        # Treat image and return binary
        sobelx = self.get_sobel(frame_gray, self.Vars['sobel_order'], 0)
        sobely = self.get_sobel(frame_gray, 0, self.Vars['sobel_order'])
        sobelBoth = cv2.bitwise_or(sobelx, sobely)
        return sobelBoth


if __name__ == '__main__':
    import detection_main

    detection_main.start_program(trackMethod='auto', mode='gradient', GUI=False)

import numpy as np  # numpy
import cv2  # OpenCV


class ThreshClass(object):
    adaptiveType = {'Gaussian average': 'cv2.ADAPTIVE_THRESH_GAUSSIAN_C', 'Normal mean': 'cv2.ADAPTIVE_THRESH_MEAN_C'}
    threshType = {False: 'cv2.THRESH_BINARY', True: 'cv2.THRESH_BINARY_INV'}

    def __init__(self):
        pass

    ############ SPECIFIC FUNCTIONS ###############

    def image_treatment(self, frame_gray):
        # Thresholding
        adType = self.adaptiveType[self.Vars['thr_adType']]
        thrType = self.threshType[self.Vars['thr_BonW']]
        thresh = cv2.adaptiveThreshold(frame_gray, 255, eval(adType), eval(thrType), self.Vars['thr_size'], self.Vars[
            'thr_C'])  # Second to last parameter controls size of the region. 11 is usually good.

        thresh = self.morphology(thresh)

        return thresh

    def binarization(self, frame, *args):
        return self.image_treatment(frame)


if __name__ == '__main__':
    import detection_main

    detection_main.start_program(trackMethod='auto', mode='thresh', GUI=False)

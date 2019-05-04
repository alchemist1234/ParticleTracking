import numpy as np  # Arrays
import cv2  # OpenCV


class AbsThreshClass(object):

    def __init__(self):
        pass

    ############ SPECIFIC FUNCTIONS ###############

    def image_treatment(self, frame_gray):

        # Thresholding
        if self.Vars['thr_BonW']:
            thr_type = 'cv2.THRESH_BINARY_INV'
        else:
            thr_type = 'cv2.THRESH_BINARY'

        ret, thresh = cv2.threshold(frame_gray, self.Vars['thr_thresh'], 255,
                                    eval(thr_type))  # Invert so black Janus particles are shown in white

        thresh = self.morphology(thresh)

        return thresh

    def binarization(self, frame, *args):
        return self.image_treatment(frame)


if __name__ == '__main__':
    import detection_main

    detection_main.start_program(trackMethod='auto', mode='absThresh', GUI=False)

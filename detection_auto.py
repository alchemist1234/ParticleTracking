import numpy as np  # Arrays
import cv2  # OpenCV
import tkinter  # GUI interface
import tkinter.ttk  # GUI interface
import detection_functions as dfun  # Tracking functions
import GUI_main  # GUI classes
import time  # For time
import matplotlib.pyplot as plt

############ SPECIFIC FUNCTIONS ###############

DetectionFunctions = dfun.DetectionFunctions


def class_factory(mode, *args, **kwargs):
    # Returns the TrackingProgram class with the appropriate binded methods defined by "mode"
    if mode == 'gradient':
        import detection_gradient
        ModeClass = detection_gradient.GradientClass

    elif mode == 'thresh':
        import detection_thresh
        ModeClass = detection_thresh.ThreshClass

    elif mode == 'absThresh':
        import detection_absThresh
        ModeClass = detection_absThresh.AbsThreshClass

    elif mode == 'bgextr':
        import detection_bgextr
        ModeClass = detection_bgextr.BgextrClass

    elif mode == 'manual':
        # Throwaway class
        class ManualModeClass:
            pass

        ModeClass = ManualModeClass

    class TrackingProgram(ModeClass, DetectionFunctions):
        # Class that performs the tracking

        GUIdict = {'gradient': GUI_main.gradientGUI, 'thresh': GUI_main.threshGUI, 'absThresh': GUI_main.absThreshGUI,
                   'bgextr': GUI_main.bgextrGUI, 'manual': GUI_main.manualTrackGUI}
        trackMethodTitle = {'auto': 'Automatic ', 'live': 'Live ', 'manual': 'Manual '}
        trackMethod = 'auto'
        trackMode = mode
        trackInfo = ['position']
        window_position = (10, 10)  # Position of the window

        #        if 'nothing' not in kwargs['extraInfo']:
        #            #*args is a list with the names of all the additional information
        trackInfo.extend(kwargs['extraInfo'])

        def __init__(self):
            pass

        def run_main(self):

            self.loopTrack = True
            while (self.loopTrack):
                ret = self.run_init()  # Function defined here

                if ret == False:
                    # Go back to main menu
                    break

                self.run_video()  # Defined here
                self.post_processing()  # Inherited from DetectionFunctions

        def run_init(self):
            ##### INITIALISING PARAMETERS #####
            # # # GUI # # #
            root = tkinter.Tk()
            self.app = self.GUIdict[self.trackMode](root, self.trackInfo, self.trackMethod)  # Start the appropriate GUI

            # self.app.root.title(self.trackMethodTitle[self.trackMethod] + self.app.windowTitle) #Put appropriate title
            root.mainloop()

            # Return to main menu?
            if self.app.loopTrack == False:
                return False

            # Import variables in dictionary
            self.Vars = {}
            for var in self.app.variables:
                self.Vars[var] = self.app.variables[var]  # .get()

            self.Vars['mode'] = self.trackMode
            self.Vars['trackMethod'] = self.trackMethod

            self.totalFramesText = ''
            if self.app.vid_totalFrames.get() != 0:
                self.totalFramesText = '/' + str(self.app.vid_totalFrames.get())

            # For choosing the video
            self.vid = cv2.VideoCapture(self.Vars['filename'])
            self.width, self.height = int(self.vid.get(3)), int(self.vid.get(4))
            self.shape = (self.width, self.height)
            FPS = self.vid.get(5)

            # Handle chosen variables
            try:
                vid_FPS = float(self.Vars['vid_FPS'])
                if vid_FPS == 0:
                    raise ValueError
            except ValueError:
                if not np.isnan(FPS):
                    vid_FPS = FPS
                else:
                    vid_FPS = 25.0

            self.Vars['vid_FPS'] = vid_FPS
            if self.Vars['mode'] != 'manual':
                vid_FPS /= float(self.Vars['vid_skip'])  # Do not divide in manual tracking mode

            try:
                self.Vars['vid_end'] = float(self.Vars['vid_end'])
            except ValueError:
                self.Vars['vid_end'] = 0

            if self.Vars['vid_end'] < 1:
                self.Vars['vid_end'] = np.inf

            # For saving the video
            path_parts = self.Vars['filename'].split('/')
            path_parts[-1] = path_parts[-1][:-4]
            self.path = '/'.join(path_parts[:-1]) + '/'
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Define codec
            filename_out = self.path + path_parts[-1] + '_nanoDetection' + '.avi'
            self.out = cv2.VideoWriter(filename_out, fourcc, vid_FPS,
                                       (self.width, self.height))  # Create VideoWriter object

            # Objects needed for tracking
            self.defunktParticles = set([])
            self.infoDict = {}  # { (particle, 'info') : [info1, info2, ...]}
            self.activeDict = {}  # { row : particle label}
            self.binary_args = []  # Additional objects used in the binarization step

            self.label = 0  # Label for the particles
            self.count = 0  # Counts frames
            self.countData = 0  # Counts analyzed frames (will vary from "count" when vid_skip > 1)
            if self.Vars['vid_fit']:
                swidth, sheight = dfun.monitor_resolution()
                self.needsFitting, self.ff = dfun.needs_fitting(self.width, self.height, swidth,
                                                                sheight)  # If it doesn't fit on screen, compute the resize factor
            else:
                self.needsFitting, self.ff = False, 1  # Will not be fitted no matter what

            self.recording = True  # Is the video being recorded?
            self.frameMode = False  # Is frame-by-frame mode activated?
            self.active = 0  # Counts number of active particles

            # Windows
            cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)  # WINDOW_NORMAL
            cv2.moveWindow('frame', 100, 100)
            try:
                if self.Vars['vid_contour']:
                    cv2.namedWindow('contour', cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow('contour', 100 + self.width, 100)
            except KeyError:
                pass  # If 'vid_contour' variable doesn't exist the program will ignore it

            if 'angleJanus' in self.trackInfo:
                self.vecDict = dfun.get_vector_circle(radi=self.Vars['extra_partRadi'],
                                                      ticks=self.Vars['extra_ang_ticks'])

            # Special variables
            self.contourMode = cv2.CHAIN_APPROX_SIMPLE
            if 'angleJetBubble' in self.trackInfo:
                # Make the contours function save all cnt points
                self.contourMode = cv2.CHAIN_APPROX_NONE

            if self.trackMode == 'bgextr':
                self.bg_reference = self.smoothing(self.app.bg_reference)[1]
                self.bg_mode = self.app.bg_mode.get()
                self.binary_args = self.bg_reference  # Will be passed to binarization function

        def run_video(self):
            # Runs the video and analyzes it

            while (True):
                self.count += 1  # Number of frames
                ret, frame = self.vid.read()

                if ret == False:
                    self.app.vid_totalFrames.set(
                        self.count - 1)  # If the end is reached, save the total number of frames
                    break

                # Stop video
                if not self.count <= self.Vars['vid_end']:
                    break

                # Skip frames
                if (self.count - self.Vars['vid_start']) % self.Vars['vid_skip'] != 0 or not self.Vars[
                                                                                                 'vid_start'] <= self.count:
                    continue
                # Save frame number
                self.countData += 1

                original = frame.copy()

                # VIDEO ANALYSIS (DETECTION, TRACKING, ETC.)
                frame, show = self.analyze_new(frame)

                # Show counter
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(self.count) + self.totalFramesText, (5, 15), font, 0.5, (255, 0, 0), 2,
                            cv2.LINE_AA)

                # Fitting in monitor
                if self.needsFitting == True:
                    frame_show = cv2.resize(frame, None, fx=self.ff, fy=self.ff, interpolation=cv2.INTER_LINEAR)
                    show = cv2.resize(show, None, fx=self.ff, fy=self.ff, interpolation=cv2.INTER_LINEAR)
                else:
                    frame_show = frame.copy()

                cv2.imshow('frame', frame_show)
                if self.countData == 1:
                    cv2.moveWindow('frame', self.window_position[0], self.window_position[1])
                if self.Vars['vid_contour']:
                    cv2.imshow('contour', show)
                    if self.countData == 1:
                        cv2.moveWindow('contour', self.window_position[0] + int(self.ff * self.width),
                                       self.window_position[1])

                ch = cv2.waitKey(1) & 0xFF

                if ch == 27:
                    # Exit
                    break
                elif ch == ord('z'):
                    # Save frame
                    cv2.imwrite(self.path + 'frame.png', original)
                elif ch == ord('p') or self.frameMode == True:  # or self.count in [11]:
                    # Pause
                    ch2 = cv2.waitKey() & 0xFF
                    if ch2 == 81 or ch2 == 83 or ch2 == 0:
                        # Next Frame
                        self.frameMode = True
                    else:
                        self.frameMode = False
                elif ch == ord('r'):
                    # Toggle resize
                    self.needsFitting = not self.needsFitting

                if self.recording == True:
                    self.out.write(frame)

                # [modified] pause 5s so that have time to choose which particle to track
                # if self.count == 2:
                #     time.sleep(5)
                # [modified] press "p" can pause the video, cancel the 5 seconds sleep

            self.out.release()
            self.vid.release()
            cv2.destroyAllWindows()

            try:
                self.Vars['totalFrames'] = int(self.app.vid_totalFrames.get())
            except ValueError:
                self.Vars['totalFrames'] = 0

        def treat_image(self, frame, *args):
            # Treat image and return the contours
            frame, frame_gray = self.smoothing(frame)
            binary = self.binarization(frame_gray, *args)
            contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, self.contourMode)

            return frame, frame_gray, binary, contours

        def get_contours(self, contours):
            # Returns list of image moments from accepted particles and modified contours. Input should be a list of ALL detected contours

            # Will store ordered current centers (-1 to distinguish unoccupied ones)
            M_list = []  # Stores the moment vectors of the selected contours
            cnt_list = []  # Stores the contours (used in bg extr mode)

            # 1ST STEP: contours are discriminated. Also special functions like particle separation are used
            for cntInd in range(len(contours)):

                self.cnt = contours[cntInd]

                # Is it a nanomotor?
                judgement, M = self.particle_test(self.cnt)

                if judgement == 'True':
                    M_list.append(M)
                    cnt_list.append(self.cnt)

                elif 'touching' in judgement:
                    # Differentiate touching particles
                    cnt_list_split = dfun.split_particles_ROI(self.Vars['extra_split_r'], self.cnt)

                    if len(cnt_list_split) > 1:
                        # If more than two particles have been found, store them
                        for cnt in cnt_list_split:
                            print(cnt)
                            M_list.append(cv2.moments(cnt))
                            cnt_list.append(cnt)


                    elif len(cnt_list_split) == 1 and "True" in judgement:
                        # If failed, consider it a single (passing) particle
                        M_list.append(M)  # M_list.append(M)     #pass
                        cnt_list.append(cnt_list_split[0])

            return M_list, cnt_list

        def analyze_new(self, frame):

            self.activeParticles = []
            self.centerDict = {}
            orderedCenters = -1 * np.ones((self.active, 2, 1))

            # 1ST STEP: contours are discriminated. Also special functions like particle separation are used
            frame, frame_gray, binary, contours = self.treat_image(frame, self.binary_args)
            show = binary.copy()
            M_list, cnt_list = self.get_contours(contours)

            # 2ND STEP: we assign the current center to the closest previous one
            for (Mind, M) in enumerate(M_list):
                center = self.get_contour_information(M, frame_gray=frame_gray, cnt=cnt_list[Mind],
                                                      binary=binary)  # Get center and store extra info if necessary

                found = False
                if self.active > 0:  # We need at least one previous position
                    diff = np.linalg.norm(self.lastCenters - center,
                                          axis=1)  # Distance from current center to all previous ones
                    closest = diff.argmin()  # Closest particle index
                    dist = diff[closest]  # Closest distance
                    if dist < self.Vars['trk_radi']:
                        # Accept it if it's within range
                        stack = np.where(orderedCenters[closest, 0, :] != -1)[
                            0].size  # Index used to stack matched centers
                        if stack > 0 and stack + 1 > orderedCenters.shape[2]:
                            # We need to add a new array along the 3rd axis
                            newarray = -1 * np.ones((orderedCenters.shape[0], 2, 1))
                            orderedCenters = np.append(orderedCenters, newarray, axis=2)

                        orderedCenters[closest, :, stack] = center
                        found = True

                if not found:
                    # Not found: must be a new one
                    newrow = -1 * np.ones((1, 2, orderedCenters.shape[2]))
                    newrow[0, :, 0] = center
                    orderedCenters = np.append(orderedCenters, newrow, axis=0)

            # 3RD STEP: handle the ordered centers list for repetitions
            maxstack = orderedCenters.shape[2]
            if maxstack > 1:
                if maxstack > 2:
                    # print 'happening'
                    pass
                # Some previous centers are matched by more than one current center
                stackIndex = np.where(orderedCenters[:, 0, -1] != -1)[0]  # Rows with stacked centers
                # print orderedCenters
                # print stackIndex

                # First the centers are sorted along the "stack" axis based on their distance to the prev center
                for rowInd in stackIndex:
                    row = orderedCenters[rowInd, :, :]
                    # print self.lastCenters[rowInd]
                    currStack = np.where(row[0, :] != -1)[0].size  # Stack of this row
                    row = row[:, :currStack]  # Stacked centers for this row
                    diff = np.linalg.norm(self.lastCenters[rowInd, :].reshape(2, 1) - row, axis=0)  # Diff with previous
                    order = np.argsort(diff).ravel()  # Order them based on distance
                    orderedCenters[rowInd, :, :currStack] = row[:, order]
                    # print orderedCenters[rowInd,:,:currStack]
                    #### THIS IS NOT COMPLETE AND DOES NOT TAKE ALL CASES INTO ACCOUNT ####
                    for counter in range(1, currStack):
                        extraCenter = orderedCenters[rowInd, :, counter]
                        # Assumption 1: the extra centers correspond to unmatched centers
                        unmatchIndex = np.where(orderedCenters[:, 0, 0] == -1)[0]  # Rows with no matched centers
                        if len(unmatchIndex) > 0:
                            diff = np.linalg.norm(self.lastCenters[unmatchIndex, :] - extraCenter, axis=1)
                            closest = diff.argmin()
                            dist = diff[closest]
                            if dist < self.Vars['trk_radi']:
                                orderedCenters[unmatchIndex[closest], :, 0] = extraCenter
                                continue

                        # If nothing was found, treat it as a new particle
                        newrow = -1 * np.ones((1, 2, orderedCenters.shape[2]))
                        newrow[0, :, 0] = extraCenter
                        orderedCenters = np.append(orderedCenters, newrow, axis=0)

                # Eliminate "stack" axis
                orderedCenters = orderedCenters[:, :, 0]  # np.delete(orderedCenters, range(1,maxstack+1), axis = 2)
                # print orderedCenters

            # 4TH STEP: all previous centers are matched, so we save the information
            #### Maybe use np.vectorize?
            for row in range(orderedCenters.shape[0]):
                center = tuple(np.ravel(orderedCenters[row, :]).astype(int))
                if -1 not in center:

                    if row < self.active:
                        # Matched center
                        self.trackedParticle = self.activeDict[row]

                    else:
                        # Unmatched center -> new particle
                        self.new_particle()
                        self.trackedParticle = self.label

                    data = [center] + [self.centerDict[center, info] for info in self.trackInfo[1:]]  # Particle info
                    self.actualize_info(self.trackedParticle, data)  # Save it
                    self.draw_frame(frame, self.trackedParticle, data)  # Draw it

                    if self.trackMode == 'bgextr' and self.bg_mode == 'Single Frame' and self.countData % self.Vars[
                        'bg_actualize'] == 0:
                        # Purge background (single frame reference mode)
                        self.reference_treatment(self.trackedParticle, frame_gray, binary)

            self.check_and_purge_particles_new()

            if self.trackMode == 'bgextr' and self.bg_mode == 'Dynamic Average':
                # Actualize reference image
                self.reference_actualize_dynamic(frame_gray)

            return frame, show

        def check_and_purge_particles_new(self):
            # Check what particles are not on frame and mark the old ones

            self.lastCenters = np.ones((0, 2), dtype=int)  # Will store previous centers
            self.activeDict = {}  # Row index : particle label
            counter = 0

            for particle in range(1, self.label + 1):
                isthere = particle in self.activeParticles

                if not isthere:
                    for info in self.trackInfo:
                        if info == 'position':
                            self.infoDict[particle, info].append((None, None))
                        else:
                            self.infoDict[particle, info].append(None)

                # Mark old particles
                if particle not in self.defunktParticles:
                    center, lastSeen = dfun.get_prev_c(self.infoDict[particle, 'position'])
                    center, lastSeen = np.array(center).reshape(1, 2), -lastSeen
                    if lastSeen > self.Vars['trk_memory']:
                        self.defunktParticles.add(particle)
                    else:

                        #                        predict_c = dfun.predict_next(self.infoDict[particle, 'position'])
                        #                        self.lastCenters = np.append(self.lastCenters, center, axis = 0)
                        self.lastCenters = np.append(self.lastCenters, center,
                                                     axis=0)  ########## PREDICTIVE BEHAVIOUR ##########
                        self.activeDict[counter] = particle
                        counter += 1
            self.active = self.lastCenters.shape[0]  # Number of non-defunkt particles


        def particle_test(self, contour, split=True):
            # In this function we test whether or not the contour corresponds to a particle
            M = cv2.moments(contour)
            area = M['m00']
            judgement = str(False)

            if area <= self.Vars['trk_maxArea'] and area > self.Vars[
                'trk_minArea']:  # Arbitrary parameters to filter contours, depends on video
                judgement = str(True)

            if self.Vars['extra_split'] == True:
                # Check if there are touching particles

                r_approx = np.sqrt(area / np.pi)
                # print r_approx, contour[0,0,:]
                is_touching = self.Vars['extra_split_r'] * 1.05 <= r_approx
                if is_touching == True:
                    # Contour might be two particles touching
                    judgement = 'touching' + str(judgement)

            return judgement, M

    return TrackingProgram  # Returns class


if __name__ == '__main__':
    import detection_main

    detection_main.start_program(trackMethod='auto')

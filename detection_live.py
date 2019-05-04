import numpy as np  # Arrays
import cv2  # OpenCV
import tkinter  # GUI interface
import tkinter.ttk  # GUI interface
import detection_functions as dfun  # Tracking functions
import detection_auto as det_auto
import GUI_main  # GUI classes


############ SPECIFIC FUNCTIONS ###############

def class_factory_live(mode, *args, **kwargs):
    TrackClass = det_auto.class_factory(mode, *args, **kwargs)

    class TrackingProgramLive(TrackClass):
        # Class that performs the live tracking (must inherit from the main class)
        trackMethod = 'live'
        ROI_r_multiplier = 1.2  # Radius of ROI will be the radius of the bounding square times this number

        def __init__(self):
            pass

        def run_init(self):
            ##### INITIALISING PARAMETERS #####

            ret = super(TrackingProgramLive, self).run_init()  # Initialize all parameters from parent class
            if ret == False:
                return ret

            # Initialize some additional parameters
            self.clickX, self.clickY = [], []
            # Mouse
            cv2.setMouseCallback('frame', self.select_contour)
            if self.Vars['vid_contour']:
                cv2.setMouseCallback('contour', self.select_contour)
            self.searching = False  # Searching contour to match the clicked point
            self.searchingCounter = 0  # Number of frames it will try to assign a contour to a clicked point
            self.analysisState = 'off'  # "off" for no analysis, 'on' for ROI analysis, 'whole' for entire frame analysis

            self.Vars['live_extra_info'] = False  # Wether or not to get extra information for un-selected contours
            if 'angleJetBubble' in self.trackInfo:
                self.Vars['live_extra_info'] = True  # We need the corrected center in this case

        def analyze_new(self, frame):

            self.activeParticles = []  # Particles that appear in this frame
            self.centerDict = {}  # { center : info}

            if self.analysisState != 'off':
                original_frame = frame.copy()

                if self.analysisState == 'whole':

                    # Treat image and return binary
                    frame, frame_gray = self.smoothing(frame)
                    binary = self.binarization(frame_gray)
                    show = binary.copy()

                    # Get janus rough contour from mask
                    self.contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                elif self.analysisState == 'on':
                    self.contours = []
                    binary = np.zeros(frame.shape[:2])  # Will be "filled" with analyzed ROI
                    frame_gray = np.zeros(frame.shape[:2], dtype=np.uint8)
                    # Analyze small sections of the image
                    for partInd, (cx, cy) in enumerate(self.lastCenters):

                        binary_args_ROI = []
                        r = self.infoDict[self.activeDict[partInd], 'ROI_r']

                        # Make sure the ROI is inside the contour of the image
                        p1 = dfun.bound((self.width, self.height), [cx - r, cy - r])  # Corner of the ROI
                        p2 = dfun.bound((self.width, self.height), [cx + r, cy + r])  # Opposite corner of the ROI
                        ROI_s = (slice(p1[1], p2[1]), slice(p1[0], p2[0]))  # ROI slice for x and y coordiantes
                        frame_ROI = original_frame[ROI_s[0], ROI_s[1]]

                        if self.trackMode == 'bgextr':
                            # Select the ROI for the reference image
                            binary_args_ROI = self.binary_args[ROI_s[0], ROI_s[1]]

                        frame_ROI, frame_gray_ROI, binary_ROI, contours_ROI = self.treat_image(frame_ROI,
                                                                                               binary_args_ROI)

                        # Add ROI information to self.contours and frame images
                        if len(contours_ROI) > 0:  # At least one contour must be found
                            contours_ROI = [cnt + p1 for cnt in
                                            contours_ROI]  # re-scale the coordinates of the contour points
                            self.contours.extend(contours_ROI)
                            binary[ROI_s[0], ROI_s[1]] = binary_ROI
                            frame_gray[ROI_s[0], ROI_s[1]] = frame_gray_ROI
                            frame[ROI_s[0], ROI_s[1]] = frame_ROI

                    show = binary.copy()

                self.matchDict = {part: [] for part in
                                  list(self.activeDict.values())}  # { particle : (contour index, distance)}

                # 1ST STEP: contours are discriminated. Also special functions like particle separation are used
                M_list, cnt_list = self.get_contours(self.contours)

                # 2ND STEP: we assign the current center to the closest previous one
                for (Mind, M) in enumerate(M_list):
                    center = self.get_contour_information(M, frame_gray=frame_gray, extra=self.Vars['live_extra_info'],
                                                          cnt=cnt_list[
                                                              Mind])  # Get center and store extra info if necessary
                    if self.activeDict:
                        diff = np.linalg.norm(self.lastCenters - center,
                                              axis=1)  # Distance from contour center to previous particle ones
                        closest = diff.argmin()  # Closest particle index
                        dist = diff[closest]  # Closest distance
                        if dist < self.Vars['trk_radi']:
                            # Match the contour with the particle
                            particle = self.activeDict[closest]
                            self.matchDict[particle].append((Mind, dist, tuple(center)))

                            # 3RD STEP: ASSOCIATE PARTICLES WITH FOUND CONTOURS
                partIter = iter(list(self.activeDict.values()))

                for particle in partIter:
                    # We get the matched center closest to the particle
                    matchedCnt = self.matchDict[particle]
                    if len(matchedCnt) > 0:
                        cntInd, dist, center = min(matchedCnt,
                                                   key=lambda a: a[1])  # Search the cntInd with least distance

                        self.save_boundingRect_ROI_r(cnt_list[cntInd],
                                                     particle)  # Save the dimensions of the bounding rectangle

                        # For efficiency we only get extra info of the selected contours
                        if len(self.trackInfo[1:]) > 0:
                            M = cv2.moments(cnt_list[cntInd])
                            self.get_contour_information(M_list[cntInd], frame_gray=frame_gray, cnt=cnt_list[cntInd])

                        data = [center] + [self.centerDict[center, info] for info in
                                           self.trackInfo[1:]]  # Particle info
                        self.actualize_info(particle, data)  # Save it
                        self.draw_frame(frame, particle, data)  # Draw it

            else:
                show = np.zeros(frame.shape[:2])

            # Check the contours clicked during this frame
            self.check_contour(frame)

            # Check particles and purge the old ones
            self.check_and_purge_particles_new()

            if self.trackMode == 'bgextr' and self.bg_mode == 'Dynamic Average':
                # Actualize reference image
                frame_gray = self.smoothing(frame)[1]
                self.reference_actualize_dynamic(frame_gray)

            if len(self.activeDict) > 0:
                self.analysisState = 'on'
            else:
                self.analysisState = 'off'

            return frame, show

        def check_contour(self, frame, *args):

            # Store previous non-defunkt particles AND newly detected ones ("set" avoids overlap)
            allParticles = set(list(self.activeDict.values()) + self.activeParticles)
            self.centerDict = {}

            # Check which contour is clicked
            if self.searching == True:

                frame, frame_gray, binary, contours = self.treat_image(frame, self.binary_args)  # Treat image
                M_list, cnt_list = self.get_contours(
                    contours)  # Get contour information, separate touching particles, etc.

                for xpoint, ypoint in zip(self.clickX, self.clickY):
                    for M, cnt in zip(M_list, cnt_list):
                        if cv2.pointPolygonTest(cnt, (xpoint, ypoint), False) > -1:
                            # Get the current center and other data
                            cx, cy = self.get_contour_information(M, frame_gray=frame_gray, cnt=cnt)

                            # Is it from an already selected center?
                            makeNew = True
                            for particle in allParticles:
                                part_center = dfun.get_prev_c(self.infoDict[particle, 'position'])[0]
                                if cv2.pointPolygonTest(cnt, tuple(part_center), False) > -1:
                                    # Stop following the particle
                                    makeNew = False
                                    self.defunktParticles.add(particle)
                                    allParticles.remove(particle)
                                    break

                            if makeNew == True:
                                # Save particle
                                self.new_particle();
                                particle = self.label
                                data = [(cx, cy)] + [self.centerDict[(cx, cy), info] for info in
                                                     self.trackInfo[1:]]  # Particle info
                                self.actualize_info(self.label, data)
                                allParticles.add(particle)
                                self.save_boundingRect_ROI_r(cnt, particle)
                            break

                    else:  # no break
                        continue
                    # If the previous loop was exited with a "break", the contour was found and the point should be removed
                    self.clickX.remove(xpoint);
                    self.clickY.remove(ypoint)

                self.searchingCounter += 1
                if self.searchingCounter == 10:
                    # After 10 frames the mouse click is ignored
                    self.clickX, self.clickY = [], []

                if len(self.clickX) == 0 and len(self.clickY) == 0:
                    # Stop searching when the lists of clicked points are cleared
                    self.searching = False

            # Optimize ROI radius for each particle
            for particle in allParticles:
                # ROI radius will be the maximum between 1.3 times the area or 1 + velocity
                # Velocity
                c2, n2 = dfun.get_prev_c(self.infoDict[particle, 'position'], num=2)
                if c2[0] != None:
                    c1, n1 = dfun.get_prev_c(self.infoDict[particle, 'position'], num=1)
                    vel = abs(
                        np.linalg.norm(np.array(c1) - np.array(c2)) / float(n2 - n1))  # Velocity is just distance/time
                #                    if n1 > 1:
                #                        vel = vel * n1
                else:
                    vel = 0

                # Area
                area_r = self.infoDict[particle, 'ROI_r']
                r = max(area_r, area_r / self.ROI_r_multiplier + vel)

                self.infoDict[particle, 'ROI_r'] = r

        def save_boundingRect_ROI_r(self, cnt, particle):
            _, _, w, h = cv2.boundingRect(cnt)
            self.infoDict[particle, 'ROI_r'] = int(max(w, h) * self.ROI_r_multiplier)

        def select_contour(self, event, x, y, flags, param):
            # clickX and clickY queue all the clicks in them
            if event == cv2.EVENT_LBUTTONUP:
                if self.needsFitting:
                    # Re-scale clicked points if the image has been scaled
                    x = x * (1 / self.ff);
                    y = y * (1 / self.ff)
                    x = int(x);
                    y = int(y)

                self.clickX.append(x);
                self.clickY.append(y)
                self.searching = True
                self.searchingCounter = 0

    return TrackingProgramLive


if __name__ == '__main__':
    import detection_main

    detection_main.start_program(trackMethod='live')

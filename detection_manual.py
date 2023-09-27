import copy  # For copying
import numpy as np  # Arrays
import cv2  # OpenCV
import tkinter  # GUI interface
import tkinter.filedialog  # GUI ask for filename
import tkinter.ttk  # GUI interface
import detection_functions as dfun  # Tracking functions
import detection_auto as det_auto
import GUI_main  # GUI classes


def class_factory_manual(mode, *args, **kwargs):
    TrackClass = det_auto.class_factory(mode, *args, **kwargs)

    class TrackingProgramManual(TrackClass):

        trackMethod = 'manual'

        def __init__(self):
            pass

        def run_init(self):

            ##### INITIALISING PARAMETERS #####
            ret = super(TrackingProgramManual, self).run_init()  # Initialize all parameters from parent class
            if ret == False:
                return ret
            # Manual Tracking variables
            self.trackInfo = ['position']
            if self.Vars['manual_angle'] == True:
                self.trackInfo.append('manual_angle')
                self.Vars['manual_angle_r'] = [0, 0]

            self.colorList = dfun.get_colors(10)
            self.colorListLen = len(self.colorList)
            self.Vars['draw_thick'] = 1
            self.new_particle()
            self.starting_particle = False  # When a particle has not started yet

            # Mouse
            cv2.namedWindow('frame')
            cv2.setMouseCallback('frame', self.handle_mouse_event)
            self.searching = 'False'

            # cv2.destroyWindow('contour')

            # Parameters needed for tracking
            self.chm = 0  # Key pressed in manual mode
            self.posCounter = -1  # Counter to know how many frames are you going back
            self.countDataMax = 0  # Maximum countData reached (might not coincide with countData after moving backwards)

            #            Frame saver: #status saves the information susceptible to change after correction, frame saves the information relative to the position of the video (not susceptible to change after correction)
            self.statusStorage = [None] * len(list(range(self.Vars[
                                                             'manual_store'])))  # [[frameLines(n), Positions(n), countData(n)], [frameLines(n-1),...], ..., [frameLines(n-100),...]}
            self.frameStorage = [None] * len(
                list(range(self.Vars['manual_store'])))  # [[frame_n, countData_n],[frame_n-1, countData_n-1]...]

            self.clickX, self.clickY = (0, 0)

        def run_video(self):

            while (True):
                if self.posCounter == -1:
                    # Normal mode: get next frame

                    self.count += 1  # Number of frames
                    ret, frame = self.vid.read()

                    if ret == False:
                        self.app.vid_totalFrames.set(
                            self.count - 1)  # If the end is reached, save the total number of frames
                        break

                    # Stop video
                    if not self.count <= self.Vars['vid_end']:
                        break

                    # Overlap current frame with tracked lines
                    frame_show, order = self.overlap_frame_show(frame)

                    if order == 'break':
                        # Exit
                        break

                    # Analyze required frames
                    if (self.count - self.Vars['vid_start']) % self.Vars['vid_skip'] == 0 and self.Vars[
                        'vid_start'] <= self.count:
                        ##### IF IT'S ONE OF THE SELECTED FRAMES, START MANUAL TRACKING MODE #####

                        # Store frame
                        self.statusStorage = dfun.store(self.statusStorage, frame)

                        # Wait for user input: track a point, create a new particle, exit, correction mode,...
                        order = self.manual_selection(frame_show, frame)

                    if self.recording == True:
                        self.out.write(self.saveframe)

                    self.countDataMax = self.countData  # Save the highest countData reached

                # Correction mode
                elif self.posCounter > -1:

                    frame = self.statusStorage[self.posCounter]
                    self.countData = self.countDataMax - self.posCounter
                    self.count = self.Vars['vid_start'] + self.countData * self.Vars['vid_skip']
                    ##### Show frame #####

                    # Correction mode: overlap frame
                    frame_show, order = self.overlap_frame_show(frame)  #

                    # Correction mode: manual tracking (no need to check count and no actualization of information)
                    order = self.manual_selection(frame_show, frame)

                    #                    order = self.handle_keystroke(chm)

                    if order in ['advance', 'move_f']:
                        # Advance position counter
                        self.posCounter -= 1

                if order == 'break':
                    break

            self.out.release()
            self.vid.release()
            cv2.destroyAllWindows()

            try:
                self.Vars['totalFrames'] = int(self.app.vid_totalFrames.get())
            except ValueError:
                self.Vars['totalFrames'] = 0

            # Purge empty particles (in case a new one is started but not clicked)
            for p in range(1, self.label + 1):
                empty = len([pos for pos in self.infoDict[p, 'position'] if pos != (None, None)]) == 0
                if empty:
                    for info in self.trackInfo:
                        del self.infoDict[p, info]

            if 'manual_angle' in self.trackInfo:
                # Change "point2" with "angle" and compute average radius
                r = 0
                r_counter = 0
                for p in range(1, self.label + 1):
                    for ind, c in enumerate(self.infoDict[p, 'position']):
                        if c != (None, None):
                            c2 = self.infoDict[p, 'manual_angle'][ind]
                            vec = np.array(c2) - np.array(c)
                            ang = np.around(dfun.ang_from_vec(vec) % 360).astype(int)
                            self.infoDict[p, 'manual_angle'][ind] = ang
                            r += np.linalg.norm(vec)
                            r_counter += 1

                if r_counter > 0:
                    self.Vars['manual_angle_r'] = int(r / float(r_counter))
                else:
                    self.Vars['manual_angle_r'] = 0

            self.countData = self.countDataMax

        def overlap_frame_show(self, frame):

            # Overlap
            frame_show = self.overlap(frame)

            cv2.imshow('frame', frame_show)

            ch = cv2.waitKey(1) & 0xFF

            order = self.handle_keystroke(ch, onlyGeneral=True)

            if order == 'screenshot':
                cv2.imwrite(self.path + 'frame.png', frame)

            return frame_show, ch

        def overlap(self, frame):
            if self.Vars['manual_show_lines'] == True:
                # Overlap trajectory lines
                frame_show = self.draw_lines(frame.copy())
            else:
                frame_show = frame

            # Show counter
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_show, str(self.count), (5, 15), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            self.saveframe = frame_show  # Frame used to save video

            # Fitting in 1366x768 monitor
            if self.needsFitting == True:
                frame_show = cv2.resize(frame_show, None, fx=self.ff, fy=self.ff, interpolation=cv2.INTER_LINEAR)

            return frame_show

        def manual_selection(self, frame_show, frame):
            # This function handles the manual selection of position, and posibly the addition of a new particle. "Exit" and "correction mode" must be handled outside the function
            self.starting_particle = len(
                [pos for pos in self.infoDict[self.label, 'position'] if pos != (None, None)]) == 0
            # Manual track!
            self.clickedDown = False  # Boolean that indicates if a position has been selected
            self.clickedUp = False
            self.activeParticles = []  # Stores active Particles (in this case, always one)
            position_saved = False
            data = []  # Will store information

            while not (self.clickedDown and self.clickedUp):
                # Keeps actualizing with current mouse position and mock line. Ends when a position is clicked, ESC is pushed or correction mode is engaged. Only the first case saves a position.

                frameTemp = frame_show.copy()  # Temporal frame with mock line

                if self.clickedDown == False:
                    # While the user has not clicked, draw mock line
                    dfun.draw_cross(frameTemp, (self.clickX, self.clickY), 3, dfun.color_invert(
                        self.colorList[self.label % self.colorListLen]))  # Cross in the center drawn in green
                    if self.starting_particle == False:  # recorded>0:
                        prev = self.infoDict[self.label, 'position'][-1]
                        frameTemp = dfun.dashed_line(frameTemp, (prev[0] * self.ff, prev[1] * self.ff),
                                                     (self.clickX, self.clickY), 5,
                                                     self.colorList[self.label % self.colorListLen],
                                                     self.Vars['draw_thick'])  # Mock line
                #
                elif self.clickedDown == True and self.Vars['manual_angle'] == True:
                    # While the user has clicked AND not released the button, draw angle (when needed)
                    point1 = data[0]
                    point1 = (int(point1[0] * self.ff), int(point1[1] * self.ff))
                    point2 = (self.clickX, self.clickY)
                    dfun.draw_arrow(frameTemp, point1, point2, np.pi / 4,
                                    dfun.color_invert(self.colorList[self.label % self.colorListLen]),
                                    self.Vars['draw_thick'])  # Mock arrow
                    if self.starting_particle == False:
                        prev = self.infoDict[self.label, 'position'][-1]
                        frameTemp = dfun.dashed_line(frameTemp, (prev[0] * self.ff, prev[1] * self.ff), point1, 5,
                                                     self.colorList[self.label % self.colorListLen],
                                                     self.Vars['draw_thick'])  # Mock line

                cv2.imshow('frame', frameTemp)
                chm = cv2.waitKey(20)  # & 0xFF

                order = self.handle_keystroke(chm)

                if order in ['break', 'correction', 'move_b', 'move_f']:
                    break

                elif order in ['toggle', 'thickUp', 'thickDown', 'color']:
                    frame_show = self.overlap(frame)  # re-draw

                # Save the selected position after clicking
                if position_saved == False and self.clickedDown == True:
                    data.append(self.clickDownXY)
                    # cv2.circle(frame_show, tuple(self.clickXY), 2, dfun.color_fix(dfun.color_invert(self.Vars['data_color'])),-1)

                    position_saved = True

            # Save the selected information when the loop exits normally
            else:  # no break

                # Increase data counter
                self.countData += 1

                if self.Vars['manual_angle']:
                    #                    point1 = self.clickDownXY
                    #                    point1 = (point1[0]*self.ff, point1[1]*self.ff)
                    point2 = (self.clickX, self.clickY)
                    #                    vec = np.array(point2) - np.array(point1)
                    #                    ang = np.around(dfun.ang_from_vec(vec) % 360).astype(int)
                    data.append(self.clickUpXY)
                #                    self.Vars['manual_angle_r'][0] += np.linalg.norm(vec) #Save radius of arrow
                #                    self.Vars['manual_angle_r'][1] += 1 #Save number of saved radius

                if self.starting_particle == True:  # recorded == 0 and self.label != 1:
                    self.label -= 1
                    self.new_particle()
                    self.starting_particle = False

                self.actualize_info(self.label, data)

            #                if self.Vars['manual_angle'] == True:
            #                    dfun.draw_arrow(self.frameLines, point1, point2, np.pi/4, dfun.color_fix(dfun.color_invert(self.Vars['data_color'])),1)
            return order

        def handle_keystroke(self, chm, onlyGeneral=False):

            order = 'none'

            if chm == -1:
                # Loop exited with no key presses
                order = 'advance'

            elif chm == 27:
                # Esc pushed: exit
                order = 'break'

            elif chm == ord('l'):
                self.Vars['manual_show_lines'] = not (self.Vars['manual_show_lines'])
                order = 'toggle'

            elif chm == ord('z'):
                # Save frame
                order = 'screenshot'

            elif chm == ord('t'):
                # Increase line thickness
                self.Vars['draw_thick'] += 1
                order = 'thickUp'

            elif chm == ord('y'):
                # Increase line thickness
                if self.Vars['draw_thick'] > 1:
                    self.Vars['draw_thick'] -= 1
                order = 'thickDown'

            elif chm == ord('c'):
                # Randomize colors
                self.colorList = dfun.get_rand_colors(self.colorListLen)
                order = 'color'

            if onlyGeneral == False:

                if chm == 8:
                    # Backspace pushed: correction
                    if not (self.countData == 0 and self.label == 1):
                        self.handle_correction()
                        order = 'correction'

                elif chm == ord('n'):
                    # New particle
                    order = 'new'
                    if self.starting_particle == False:
                        self.starting_particle = True
                        self.new_particle()

                elif chm == ord('q'):
                    if self.starting_particle and self.countData > 0:
                        self.handle_move_backwards()
                        order = 'move_b'

                elif chm == ord('w'):
                    if self.starting_particle:
                        self.handle_move_forwards()
                        order = 'move_f'

            return order

        def handle_correction(self):
            # Backspace pushed: change
            self.posCounter = self.diminish_posCounter(self.posCounter)

            posLength = len([pos for pos in self.infoDict[self.label, 'position'] if pos != (None, None)])
            if posLength >= 1:
                # Delete data
                for info in self.trackInfo:
                    del self.infoDict[self.label, info][-1]

            # If we are canceling a new particle, modify label
            if posLength <= 1 and self.label != 1:
                if (self.label, 'position') in self.infoDict:
                    for info in self.trackInfo:
                        del self.infoDict[
                            self.label, 'position']  # We need to delete it manually because it won't be overwritten otherwise
                self.label -= 1
                newpos = self.infoDict[self.label, 'position']
                self.posCounter = self.countDataMax - len(newpos)  # Jump to the last position of the previous particle

        def handle_move_backwards(self):
            # Change posCounter
            self.posCounter = self.diminish_posCounter(self.posCounter)

        def handle_move_forwards(self):
            # Change posCounter
            self.countData += 1

        def diminish_posCounter(self, posCounter):
            if posCounter != self.Vars['manual_store'] - 1 and posCounter != self.countDataMax:
                if posCounter != -1:
                    posCounter += 1  # Go back one data point (up to the maximum stored)
                else:
                    posCounter = 1  # First time it's clicked it should be moved from -1 to 1
            return posCounter

        def handle_mouse_event(self, event, x, y, flags, param):

            if event == cv2.EVENT_LBUTTONDOWN:
                self.clickX, self.clickY = x, y
                self.clickDownXY = (
                int(np.around(self.clickX / self.ff)), int(np.around(self.clickY / self.ff)))  # Correct for resizing
                self.clickedDown = True

            elif event == cv2.EVENT_LBUTTONUP:
                self.clickX, self.clickY = x, y
                self.clickUpXY = (
                int(np.around(self.clickX / self.ff)), int(np.around(self.clickY / self.ff)))  # Correct for resizing
                self.clickedUp = True

            elif event == cv2.EVENT_MOUSEMOVE:
                self.clickX, self.clickY = x, y

        def draw_lines(self, frame):
            # Draw lines
            for p in range(1, self.label + 1):
                pos = iter(self.infoDict[p, 'position'])
                start = False
                counter = -1
                while True:
                    try:
                        c = next(pos)
                        counter += 1
                    except StopIteration:
                        break

                    if c[0] != None:
                        if start == True:
                            cv2.line(frame, c_prev, c, self.colorList[p % self.colorListLen], self.Vars['draw_thick'])
                        else:
                            start = True

                        cv2.circle(frame, c, self.Vars['draw_thick'] + 1,
                                   dfun.color_invert(self.colorList[p % self.colorListLen]), -1)
                        if self.Vars['manual_angle'] == True:
                            point2 = self.infoDict[p, 'manual_angle'][counter]
                            dfun.draw_arrow(frame, c, point2, np.pi / 4,
                                            dfun.color_invert(self.colorList[p % self.colorListLen]),
                                            self.Vars['draw_thick'])

                        c_prev = c

            return frame

    return TrackingProgramManual


#
############# SPECIFIC FUNCTIONS ###############
#
# def handle_mouse_event(event, x, y, flags, param):
#
#    if event == cv2.EVENT_LBUTTONUP:
#        clickX, clickY = x, y
#        clicked = True
#
#    elif event == cv2.EVENT_MOUSEMOVE:
#        clickX, clickY = x, y
#        
# def run_main():
#    global Vars, clickX, clickY, clicked, label
#
#    #Mode must be "thresh" or "gradient"
#
#    ##### INITIALISING PARAMETERS #####
#
#    #Mouse
#    cv2.namedWindow('frame')
#    cv2.setMouseCallback('frame', handle_mouse_event)
#    searching = 'False'
#
#    # # # GUI # # #
#
#    root = Tkinter.Tk()
#    app = GUI_main.manualTrackGUI(root)
#    root.mainloop()
#
#    #Import variables in dictionary
#    Vars = {}
#    for var in app.variables:
#        Vars[var] = app.variables[var].get()
#
#    #For choosing the video
#    vid = cv2.VideoCapture(Vars['filename'])
#    width,height = int(vid.get(3)),int(vid.get(4))
#    shape = (width,height)
#    FPS = vid.get(5)
#
#    #Handle chosen variables
#    try:
#        vid_FPS = float(Vars['vid_FPS'])
#        if vid_FPS == 0:
#            raise ValueError
#    except ValueError:
#        if not np.isnan(FPS):
#            vid_FPS = FPS
#        else:
#            vid_FPS = 15.0
#    #vid_FPS /= float(Vars['vid_skip'])
#    Vars['vid_FPS'] = vid_FPS
#
#    try:
#        Vars['vid_end'] = float(Vars['vid_end'])
#    except ValueError:
#        Vars['vid_end'] = 0
#
#
#    #For saving the video
#    path_parts = Vars['filename'].split('/')
#    path = '/'.join(path_parts[:-1]) + '/'
#    fourcc = cv2.VideoWriter_fourcc(*'MJPG') #Define codec
#    out = cv2.VideoWriter(path+'nanoDetection_' + path_parts[-1],fourcc,vid_FPS,(width,height)) #Create VideoWriter object
#
#    #Objects needed for tracking
#    Positions = {} # { particle : [(x0,y0),(x1,y1),...,(xf,yf)] }
#    label = 1
#    Positions[label] = [] #We initiate the particle
#    frameLines = np.zeros((height, width, 3), dtype = np.uint8) #Frame saving the drawn trajectory
#
#    chm = 0 #Key pressed in manual mode
#    count = 0 #Counts frames
#    countData = 0 #Counts analyzed frames (will vary from "count" when vid_skip > 1)
#    avNum = 10 #Averages for vectorial velocity (takes 2*avNum+1 samples)
#    boo, ff = dfun.needs_fitting(width, height, 1366, 768)
#    needsFitting = (boo and Vars['vid_fit'])
#    recording = True
#    frameMode = False
#    correctionMode = False #Is the program going through previous frames
#    posCounter = 0 #Counter to know how many frames are you going back
#
#    #Frame saver
#    statusStorage = {num:[None] for num in range(Vars['manual_store'])} #{1: [frame, frameLines, Positions, countData], 2: [...], ..., 10: [...]}
#
#    if Vars['vid_end'] < 1:
#        Vars['vid_end'] = np.inf
#
#    #Vars['data_colorB'], Vars['data_colorG'], Vars['data_colorR'] = (250,250,0)
#    data_color = [Vars['data_colorB'], Vars['data_colorG'], Vars['data_colorR']]
#    data_color = dfun.color_fix(data_color) #We fix it so that we can use it in np.where function
#    Vars['data_color'] = data_color
#
#    clickX, clickY = (0,0)
#
#    while(True):
#
#        if not correctionMode:
#            #Normal mode: get next frame
#
#            count += 1 #Number of frames
#            ret, frame = vid.read()
#            if ret == False or count > Vars['vid_end']:
#                break
#
#            ##### Show frame #####
#
#            #Overlap current frame with tracked lines
#            frame_show, ch = overlap_frame(frame, frameLines, count, needsFitting)
#            
#            if ch == 27:
#                #Exit
#                break
#                
#            ##### IF IT'S ONE OF THE SELECTED FRAMES, START MANUAL TRACKING MODE #####
#
#            if (count % Vars['vid_skip'] == 0 and count >= Vars['vid_start']):
#
#                #Store current status for posible correction
#                status = copy.deepcopy([frame, frameLines, Positions, countData])
#                #Push new data
#                statusStorage = dfun.store(statusStorage, status)
#
#                #Wait for user input: track a point, create a new particle, exit, correction mode,...
#                frameLines, Positions, countData, chm = manual_selection(frame_show, frameLines, Positions, countData)
#
#                if chm == 27:
#                    #Esc pushed: exit
#                    break
#
#                elif chm == 8:
#                    #Backspace pushed: correction
#                    correctionMode = True
#                    if posCounter != Vars['manual_store']-1 and (posCounter+1)*Vars['vid_skip'] != count:
#                        posCounter += 1 #Go back one data point (up to the maximum stored)
#
#                    #If we are canceling a new particle, modify label
#                    posLength = len( [pos for pos in Positions.values()[-1] if pos != (None, None)])
#                    if posLength == 1 and label != 1:
#                        label -= 1
#
#        #CORRECTION MODE
#        else:
#            #Correction mode: get previous status
#            frame, frameLines, Positions, countData = copy.deepcopy(statusStorage[posCounter])
#            count = (countData+1) * Vars['vid_skip']
#
#            ##### Show frame #####
#
#            #Correction mode: overlap frame
#            frame_show, ch = overlap_frame(frame, frameLines, count, needsFitting)
#            
#            #Correction mode: manual tracking (no need to check count and no actualization of information)
#            frameLines, Positions, countData, chm = manual_selection(frame_show, frameLines, Positions, countData)
#
#            if clicked:
#                #Correction mode: handle modification of information
#
#                #Advance position counter
#                posCounter -= 1
#
#                if posCounter != -1:
#                    #Modify status with corrected info (we don't need to copy the frame because it's the same)
#                    statusCorrection = [frameLines, Positions, countData]
#                    statusStorage[posCounter][-3:] = statusCorrection
#                else:
#                    #Last position has been corrected: return to normal mode (data will be saved in the "normal" while loop)
#                    posCounter = 0
#                    correctionMode = False
#
#            elif chm == 27:
#                #Exit
#                break
#
#            elif chm == 8:
#                #Go back one more step (up to the maximum stored)
#                if posCounter != Vars['manual_store']-1:
#                    posCounter += 1 
#
#                #If we are canceling a new particle, modify label
#                posLength = len( [pos for pos in Positions.values()[-1] if pos != (None, None)])
#                if posLength == 1 and label != 1:
#                    label -= 1
#
#
#        if recording == True:
#            out.write(frame)
#
#    out.release()
#    vid.release()
#    cv2.destroyAllWindows()
#
#    #Extend particle's positions
#    for particle in Positions.keys():
#        length = len(Positions[particle])
#        if length < countData:
#            Positions[particle].extend([(None,None)]*(countData-length))
#
#
#    #Cheap hack to make post-processing work
#    Vars['trackMethod'] = 'manual'
#    PostProcessing = dfun.DetectionFunctions() #Instance of the DetectionFunctions class, which has post processing
#    PostProcessing.Vars = Vars
#    PostProcessing.infoDict = {(particle,'position'): Positions[particle] for particle in Positions}
#    PostProcessing.countData = countData
#    PostProcessing.trackInfo = ['position']
#
#    PostProcessing.post_processing()
#
# def overlap_frame(frame, frameLines, count, needsFitting):
#    #Overlap trajectory lines
#    frame_show = np.where(frameLines, frameLines, frame)
#    #Show counter
#    font = cv2.FONT_HERSHEY_SIMPLEX
#    cv2.putText(frame_show,str(count),(5,15),font,0.5,(255,0,0),2,cv2.LINE_AA)
#
#    #Fitting in 1366x768 monitor
#    if needsFitting == True:
#        frame_show = cv2.resize(frame_show,None,fx = ff, fy = ff, interpolation = cv2.INTER_LINEAR)
#        show = cv2.resize(show,None,fx = ff, fy = ff, interpolation = cv2.INTER_LINEAR)
#
#    cv2.imshow('frame',frame_show)
#
#    ch = cv2.waitKey(1) & 0xFF
#
#    if ch == 27:
#        #Exit
#        pass
#    elif ch == ord('z'):
#        #Save frame
#        cv2.imwrite(path + 'frame.png',frame)
#    elif ch == ord('r'):
#        #Toggle resize
#        needsFitting = not needsFitting
#
#    return frame_show, ch
#
#
# def manual_selection(frame_show, frameLines, Positions, countData):
#    global clicked, label
#    #This function handles the manual selection of position, and posibly the addition of a new particle. "Exit" and "correction mode" must be handled outside the function
#
#    #Manual track!
#    clicked = False #Boolean that indicates if a position has been selected
#
#    while not clicked:
#        #Keeps actualizing with current mouse position and mock line. Ends when a position is clicked, ESC is pushed or correction mode is engaged. Only the first case saves a position.
#
#        frameTemp = frame_show.copy() #Temporal friend with mock line
#        dfun.draw_cross(frameTemp, (clickX, clickY), 3, (0,255,0)) #Cross in the center drawn in green
#        recorded = dfun.my_index(Positions[label][::-1],(None,None)) #Amount of centers in current particle
#
#        if recorded>0:
#            frameTemp = dfun.dashed_line(frameTemp, Positions[label][-1], (clickX, clickY), 5, Vars['data_color'],1) #Mock line
#
#        cv2.imshow('frame', frameTemp)
#        chm = cv2.waitKey(20) & 0xFF
#
#        if chm == 27:
#            #Exit
#            break
#        elif chm == 8:
#            #Correction mode
#            if countData > 0:
#                break
#        elif chm == ord('n'):
#            #New particle
#            label, Positions = dfun.new_particle(label, Positions, countData+1)
#
#    #Save the selected position 
#    if clicked:
#        #If the loop was exited after clicking
#        Positions[label].append((clickX,clickY))
#        cv2.circle(frameLines, Positions[label][-1], 2, dfun.color_fix(dfun.color_invert(Vars['data_color'])),-1)
#
#        if recorded+1 > 1:
#            cv2.line(frameLines, Positions[label][-2], Positions[label][-1], Vars['data_color'], 1)
#
#        #Increase data counter
#        countData += 1
#
#    return frameLines, Positions, countData, chm


if __name__ == '__main__':
    import detection_main

    detection_main.start_program(trackMethod='manual', GUI=False)

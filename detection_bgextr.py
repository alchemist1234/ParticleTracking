import numpy as np  # Arrays
import cv2  # OpenCV
import detection_functions as dfun  # Tracking functions


class BgextrClass(object):

    def __init__(self):
        pass

    ############ SPECIFIC FUNCTIONS ###############

    def image_treatment(self, frame_gray, *args):

        # print len(args)
        ref = args[0]

        # #Subtract reference
        subtract = frame_gray.astype(int) - ref  # self.bg_reference.astype(int)
        subtract = np.uint8(abs(subtract))

        # Threshold
        ret, thresh = cv2.threshold(subtract, self.Vars['bg_thresh'], 255, cv2.THRESH_BINARY)
        thresh = self.morphology(thresh)

        #        #Optional: show reference
        #        cv2.imshow('ref',self.bg_reference)
        #        cv2.waitKey(1) & 0xFF

        return thresh

    def reference_actualize_dynamic(self, frame_gray):
        # Actualizes the reference using the dynamic formula
        alpha = self.Vars['bg_alpha']
        ref = alpha * self.bg_reference.astype(float) + (1 - alpha) * frame_gray.astype(float)
        ref = ref.astype(np.uint8)
        self.bg_reference = ref

    def reference_treatment(self, particle, frame_gray, binary):
        #### SPECIAL BACKGROUND EXTRACTION ####
        # We differentiate real motors from artificial reflections, which appear due to the using of the first frame as reference
        # The reference is "patched" if with the current image if it detects a false nanomotor
        pos = self.infoDict[particle, 'position']

        timespan = self.Vars['bg_actualize']
        pos = [center for center in pos if center[0] != None]
        Apos = np.vstack(list(zip(*pos[-timespan:])))
        posSD = np.sum(
            np.average((Apos - np.average(Apos, axis=1).reshape(2, 1)) ** 2, axis=1))  # Standard deviation of x and y

        # First test: check standard deviation of the movement for the last 50 frames
        if posSD < 5:
            # Get points inside the contour
            nz = list(zip(*np.nonzero(self.binary)))
            cnt = self.centerDict[pos[-1], 'cnt']
            ptsContour = [point for point in nz if cv2.pointPolygonTest(cnt, point[::-1], False) >= 0]
            ptsContour = list(zip(*ptsContour))
            pixelContour = frame_gray[ptsContour]
            # Second test: check grayscale mean. Background should be whiter
            grayMean = np.average(pixelContour)
            if grayMean > self.Vars['bg_bgColor']:  # IsItBackground
                self.bg_reference[ptsContour] = frame_gray[ptsContour]

    def binarization(self, frame, *args):
        return self.image_treatment(frame, *args)


if __name__ == '__main__':
    import detection_main

    detection_main.start_program(trackMethod='auto', mode='bgextr', GUI=False)

# def post_processing(Positions, color = False):
#     skipFrames = Vars['vid_skip']
#     countData = len(Positions.values()[-1]) #The -1 is so that it works with the manual tracking script
#     Frames = list(np.arange(1,countData+1)*skipFrames)

#     #### Choose particles to analyze ####

#     root = Tkinter.Tk()
#     app2 = GUI_main.dataAnalysisGUI(root)

#     #Optionally start with a certain color
#     if color:
#         app2.data_colorB.set(color[0])
#         app2.data_colorG.set(color[1])
#         app2.data_colorR.set(color[2])
#     root.mainloop()

#     #Import variables
#     for var in app2.variables:
#         Vars[var] = app2.variables[var].get()


#     print 'Analyzing...'

#     #Time units
#     try:
#         data_FPS = float(Vars['data_FPS'])
#         unit_time = 'seconds'
#         if data_FPS == 0.:
#             raise ValueError
#     except ValueError:
#         data_FPS = 1
#         unit_time = 'frames'
#     Frames = list(np.array(Frames)/data_FPS)

#     #Distance units
#     try:
#         data_pix2mtr = float(Vars['data_pix2mtr'])
#         unit_distance = 'microm'
#         if data_pix2mtr == 0.:
#             raise ValueError
#     except ValueError:
#         data_pix2mtr = 1
#         unit_distance = 'pixels'

#     #### Purge the particles that have not been selected ####

#     if len(Vars['data_ParticleList']) >= 1:
#         data_ParticleList = [int(num) for num in Vars['data_ParticleList'].split(',')]
#         for particle in Positions.keys():
#             if particle not in data_ParticleList:
#                 del Positions[particle]

#     StartEnd = get_startend(Positions)

#     #### Saving data ####
#     path_parts = Vars['filename'].split('/')
#     path = '/'.join(path_parts[:-1]) + '/'
#     save_data_raw(path+'data_raw_' + path_parts[-1][:-4] + '.csv', Positions, Frames, unit_distance, unit_time)

#     #### Interpolate missing points with cubic spline for automatic MSD ####
#     interpMode = Vars['data_interp']
#     Positions = fill_in(Positions,StartEnd,interpMode)
#     if Vars['data_draw'] == True:
#         drawPositions = Positions.copy()

#     #Convert units
#     for particle in Positions.keys():
#         pos = Positions[particle]
#         Positions[particle] = zip(*zip(*(np.array(pos)/data_pix2mtr)))

#     #### Centered derivative to get velocity ####

#     VelVec, VelMod, VelTotal = get_velocity(Positions, Vars['data_order'])

#     #### Saving data ####
#     with open(path+'data_inter_' + path_parts[-1][:-4] + '.csv','wb') as csvfile:
#         writer = csv.writer(csvfile)

#         for particle in Positions:
#             X, Y = zip(*Positions[particle])
#             writer.writerow(['Particle',str(particle)])
#             indexes = (StartEnd[particle][0],StartEnd[particle][1]+1)
#             timeUnits = Frames[indexes[0]:indexes[1]]
#             writer.writerow(['Time ('+unit_time+')']+timeUnits)
#             writer.writerow(['X ('+unit_distance+')']+list(X))
#             writer.writerow(['Y ('+unit_distance+')']+list(Y))
#             incT, MSD = mean_square_displacement(X,Y)
#             #incT = [i/float(FPS) for i in incT]
#             writer.writerow(['Time displacement ('+unit_time+')']+list(skipFrames*incT/data_FPS))
#             writer.writerow(['MSD ('+unit_distance+'**2)']+list(MSD))
#             mod = data_FPS*VelMod[particle]/skipFrames
#             writer.writerow(['Velocity Modulus ('+unit_distance+'/'+unit_time+')'] + list(mod))
#             modAv = np.average(mod)
#             writer.writerow(['Average vel modulus',str(modAv)])
#             velTot = VelTotal[particle]
#             velTot = velTot * (data_FPS/skipFrames)
#             writer.writerow(['Total distance / Total time ('+unit_distance+'/'+unit_time+')', str(velTot)])
#             writer.writerow('')

#         sorted_keys = sorted(Vars.keys())
#         sorted_values = [Vars[key] for key in sorted_keys]
#         writer.writerow(['Parameter name:'] + sorted_keys)
#         writer.writerow(['Parameter value:'] + sorted_values)
#         writer.writerow('')

#     print 'Done'

#     #### Draw trajectories ####
#     if Vars['data_draw'] == True:

#         data_color = [Vars['data_colorB'], Vars['data_colorG'], Vars['data_colorR']]
#         for i in range(3):
#             #Cheap hack so that it's possible to use the "np.where" function
#             if data_color[i] == 0:
#                 data_color[i] = 1

#         #For choosing the video
#         vid = cv2.VideoCapture(Vars['filename'])
#         width,height = int(vid.get(3)),int(vid.get(4))
#         shape = (width,height)

#         #For saving the video
#         path_parts = Vars['filename'].split('/')
#         path = '/'.join(path_parts[:-1]) + '/'
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG') #Define codec
#         out = cv2.VideoWriter(path+'nanoDetection_marker_' + path_parts[-1],fourcc,Vars['vid_FPS'],(width,height)) #Create VideoWriter object

#         #For showing the video
#         boo, ff = needs_fitting(width, height, 1366, 768)
#         needsFitting = (boo and Vars['vid_fit'])

#         #Variables
#         prev_count = countData #Previous data count
#         count = 0 #Counts frames
#         countData = 0 #Counts analyzed frames (will vary from "count" when vid_skip > 1)
#         recording = True
#         frameMode = False
#         imgTail = np.zeros((height, width,3), dtype = np.uint8) #Blank image used to save the trajectories before overlapping to the current frame

#         #Position iterators
#         PosIter = {}
#         for particle in Positions:
#             #The iterator will go through position indexes
#             PosIter[particle] = iter(range(len(drawPositions[particle])))

#         while(True):
#             count += 1 #Number of frames
#             ret, frame = vid.read()
#             if ret == False:
#                 break

#             #Skip frames
#             if not (Vars['vid_start'] <= count <= Vars['vid_end']):
#                 continue
#             elif countData == prev_count:
#                 break

#             if count % skipFrames == 0:
#                 #Actualize the trajectory lines
#                 #Save frame number
#                 countData += 1
#                 for particle in drawPositions:
#                     if particle == 3:
#                         data_color = [155,200,50]
#                     elif particle == 7:
#                         data_color = [175,255,125]
#                         if 3 < countData < 20:
#                             posIndex = PosIter[particle].next()
#                             continue
#                     elif particle == 15:
#                         data_color = [255,195,120]


#                     start, end = StartEnd[particle]
#                     if start <= countData-1 <= end:
#                         posIndex = PosIter[particle].next()
#                         if start == countData-1:
#                             point1 = point2 = drawPositions[particle][posIndex]
#                         else:
#                             point1 = drawPositions[particle][posIndex-1]
#                             point2 = drawPositions[particle][posIndex]

#                         cv2.line(imgTail, point1, point2, data_color, 2)

#                         # # Draw label
#                         # pt = point2
#                         # font = cv2.FONT_HERSHEY_SIMPLEX
#                         # cv2.putText(frame,str(particle),(pt[0]+15,pt[1]+15),font,0.5,(0,0,255),2,cv2.LINE_AA)
#                         # cv2.circle(frame,tuple(pt),1,(0,255,0),-1)

#             #Overlap lines on frame
#             frame = np.where(imgTail, imgTail, frame)

#             #We draw the center now so that it overlaps the lines
#             for particle in drawPositions:
#                 start, end = StartEnd[particle]
#                 if start <= countData-1 < end:
#                     pt = drawPositions[particle][countData-start-1]
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     #cv2.putText(frame,str(particle),(pt[0]+15,pt[1]+15),font,0.5,(0,0,255),2,cv2.LINE_AA)
#                     cv2.circle(frame,tuple(pt),2,(0,255,0),-1)


#             #Show counter
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             cv2.putText(frame,str(count),(5,15),font,0.5,(255,0,0),2,cv2.LINE_AA)

#             #Fitting in 1366x768 monitor
#             if needsFitting == True:
#                 notFittedFrame = frame.copy()
#                 frame_show = cv2.resize(frame,None,fx = ff, fy = ff, interpolation = cv2.INTER_LINEAR)
#             else:
#                 frame_show = frame.copy()

#             cv2.imshow('frame',frame)

#             ch = cv2.waitKey(1) & 0xFF

#             if ch == 27:
#                 #Exit
#                 break
#             elif ch == ord('p') or frameMode == True:
#                 #Pause
#                 ch2 = cv2.waitKey() & 0xFF
#                 if ch2 == 81 or ch2 == 83:
#                     #Next Frame
#                     frameMode = True
#                 else:
#                     frameMode = False

#             if recording == True:
#                 out.write(frame)

#         out.release()
#         vid.release()
#         cv2.destroyAllWindows()

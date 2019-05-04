from sys import exit  # To kill process
import random  # For random numbers and selections
import time  # For time measurements
import os  # For directory finding
import csv  # For excel writing
import copy  # For dictionary deep copying
import numpy as np  # Arrays
import matplotlib.pyplot as plt  # Plotting
from scipy import ndimage  # Image processing
from scipy.interpolate import interp1d
import cv2  # OpenCV
import tkinter  # GUI interface
import tkinter.filedialog  # GUI ask for filename
import tkinter.ttk  # GUI interface
import detection_functions as dfun  # Tracking functions
import GUI_main  # GUI classes

############ SPECIFIC FUNCTIONS ###############

# Additional star variables

additional_vars = ['starRadi', 'ticks', 'img_open', 'img_close', 'sobel_thresh', 'img_gaussblur', 'img_medianblur',
                   'img_bilateral_d', 'img_bilateral_color', 'img_bilateral_space', 'power']

Vars2 = {}
Vars2['starRadi'] = 25
Vars2['ticks'] = 15
Vars2['img_open'] = 1
Vars2['img_close'] = 1
Vars2['sobel_thresh'] = 380
Vars2['img_gaussblur'] = 3
Vars2['img_medianblur'] = 3
Vars2['img_bilateral_d'] = 5
Vars2['img_bilateral_color'] = 75
Vars2['img_bilateral_space'] = 75
Vars2['power'] = 8.0


def binarization(frame, mode):
    # Binarizes the image depending on the method selected

    if mode == 'thresh':
        detection_thresh.Vars = Vars
        return detection_thresh.thresh_binarization(frame)

    elif mode == 'gradient':
        detection_gradient.Vars = Vars
        return detection_gradient.gradient_binarization(frame)

    elif mode == 'absThresh':
        detection_absThresh.Vars = Vars
        return detection_absThresh.thresh_binarization(frame)


def particle_test(contour):
    # In this function we test wether or not the contour corresponds to a particle
    M = cv2.moments(contour)
    area = M['m00']
    judgement = False

    if area <= Vars['trk_maxArea'] and area > Vars[
        'trk_minArea']:  # Arbitrary parameters to filter contours, depends on video
        judgement = True
    return judgement, M


def perpendicular(vec):
    # Returns the perpendicular of a vector
    return np.array([-vec[1], vec[0]])


def vec_from_angle(angle):
    # Returns a unitary vector at the specific angle in degrees
    angle = dfun.deg2rad(angle)
    return np.array([np.cos(angle), np.sin(angle)])


def get_rotating_vectors(num):
    # Get pairs of rotating vectors in "num" different angles between 0 and 360 degrees
    angles_bin = np.linspace(0, np.pi / 2, num + 1)[:-1]
    vec = np.array([1, 0])
    varray = np.empty((2, num))
    for n, angle in enumerate(angles_bin):
        vrot = rotate(vec, angle)
        varray[:, n] = vrot
    return varray


def get_line_points(start, end):
    # Return all the points in a line going from start to end
    dist = int(round(np.linalg.norm(end - start)))
    x = np.linspace(start[0], end[0], dist)
    y = np.linspace(start[1], end[1], dist)
    x, y = np.around(x).astype(int), np.around(y).astype(int)
    return [y, x]


def rotating_clock(picture, center, point1):
    # Gets the distance of the averaged point in the directions given by point1 and point2 calculated from the center in the four axis
    picShape = picture.shape[::-1]
    point2 = center + perpendicular(point1 - center)
    point3 = 2 * center - point1
    point4 = 2 * center - point2
    pointList = [point1, point2, point3, point4]
    distTot = 0  # It will measure d1**2 + d2**2 + d3**2 + d4**2
    for point in pointList:
        point = dfun.is_out_of_bounds(picShape, point)  # Check if it's inside
        along = get_line_points(center, point)  # Coordinates of points
        values = picture[along]  # Pixel values of points
        if np.any(values):  # It must find some points alongside the vectors
            along = np.array(along)
            along_relative = along - center[::-1].reshape(2, 1).astype(float)
            sign = np.sign(along_relative[:, -1])
            # av = np.dot(along_relative,values)/float(sum(values))
            power = Vars2['power']
            av = np.power(np.dot(along_relative ** power, values) / float(sum(values)), 1 / power)
            av = np.around(av) * sign
            # frame[av[0]+center[1],av[1]+center[0]] = (255,0,255)
            # show[av[0],av[1]] = 125
            dist = np.linalg.norm(av)
            distTot += dist ** 2
    return distTot


def rotate(point, angle):
    # Rotate the vector in an angle specified in radians
    sin = np.sin(angle)
    cos = np.cos(angle)
    rotMatrix = np.array([[cos, -sin], [sin, cos]])
    point = np.array(point)
    return np.dot(rotMatrix, point)


def get_star_angles(maskStar, center, ticks, varray, starRadi):
    # ROTATING CLOCK
    valueList = []
    for number in range(ticks):
        vec = (starRadi * varray[:, number]).astype(int)
        value = rotating_clock(maskStar, center, center + vec)
        valueList.append(value)

    ind = valueList.index(max(valueList))
    v1 = (starRadi * varray[:, ind]).astype(int)
    v2 = perpendicular(v1)
    ratioInfo = get_radius_ratio(maskStar, center, v1)
    ang1 = np.arctan2(-v1[1], v1[0]) % (2 * np.pi)
    ang2 = np.arctan2(-v2[1], v2[0]) % (2 * np.pi)
    ang3 = (ang1 + np.pi) % (2 * np.pi)
    ang4 = (ang2 + np.pi) % (2 * np.pi)
    currentAngles = dfun.rad2deg(np.vstack((ang1, ang2, ang3, ang4))).astype(int)
    return currentAngles, v1, ratioInfo


def get_radius_ratio(picture, center, vecArm):
    # Returns the average length of the arms, the average length of the corner (assumed to be at a 45 degree angle) and the ratio between them
    listArm = [];
    listCorner = []
    picShape = picture.shape[::-1]

    drawn = False

    for i in range(4):
        # We average over all 4 arms
        pointArm = dfun.is_out_of_bounds(picShape, center + vecArm)  # Check if it's inside
        (lengthArm, ptArm) = get_radius_in_direction(picture, center, pointArm)

        vecCorner = np.around(rotate(vecArm, np.pi / 4.0)).astype(int)
        pointCorner = dfun.is_out_of_bounds(picShape, center + vecCorner)
        (lengthCorner, ptCorner) = get_radius_in_direction(picture, center, pointCorner)

        if lengthArm != False and lengthCorner != False:
            listArm.append(lengthArm)
            listCorner.append(lengthCorner)

            # if drawn == False:
            #     dfun.draw_arrow(frame, tuple(center), tuple(ptArm[::-1]), np.pi/4.0, (0,0,255), 1)
            #     dfun.draw_arrow(frame, tuple(center), tuple(ptCorner[::-1]), np.pi/4.0, (0,255,255), 1)
            #     drawn = True

            cv2.line(frame, tuple(center), tuple(ptArm[::-1]), (0, 0, 255), 1)
            cv2.line(frame, tuple(center), tuple(ptCorner[::-1]), (0, 255, 255), 1)

        else:
            # print 'hey-oh'
            pass

        vecArm = rotate(vecArm, np.pi / 2)

    lengthArm = np.average(listArm);
    lengthCorner = np.average(listCorner)
    if lengthCorner != 0:
        ratio = lengthArm / lengthCorner
    else:
        print(lengthCorner)
        ratio = 0

    return lengthArm, lengthCorner, ratio


def get_radius_in_direction(picture, center, point):
    # Returns an average of the occupied points in a line between center and point
    along = get_line_points(center, point)
    values = picture[along]
    if np.any(values):
        along = np.array(along)
        av = np.dot(along, values) / float(sum(values))
        av = np.around(av).astype(int)
        # frame[av[0],av[1]] = (255,0,255)
        return (np.linalg.norm(av[::-1] - center), av)

    else:
        return False, False

    # #Gets the last occupied point in picture in the direction center -> point
    # along = get_line_points(center, point)
    # along = iter(zip(along[0],along[1])[::-1])
    # for pt in along:
    #     if picture[pt] == 255:
    #         break
    # else:
    #     return False, False
    # frame[pt[0],pt[1]] = (255,0,255)
    # return (np.linalg.norm(pt[::-1]-center), pt)


def track_angle(prev, current):
    # Sort the current angles so that they are closest to the previous ones
    extended_prev = np.hstack((prev, prev + 360, prev + 720))  # Extend the quadrants for 360->0 jumps
    current_sorted = np.ones((4, 1)) * -1
    for i in range(4):
        minInd = abs(extended_prev - (current[i] + 360)).argmin()
        axis = np.where(prev == (extended_prev.flatten()[minInd] % 360))
        current_sorted[axis] = current[i]

    if -1 in current_sorted:
        # If there's a 45 angle difference between prev and current the order is not unique and can cause problems
        # We impose a different ordering from lowest to highest. Eg: prev=[225,0,90,180], curr=[315,225,45,135], curr_sort=[315,45,135,225]
        prevf = np.sort(prev, axis=None)
        currentf = np.sort(current, axis=None)
        for i in range(4):
            ind = np.where(prev == prevf[i])
            current_sorted[ind] = currentf[i]

    return current_sorted


def get_prev_ang(angles):
    previous = np.arange(4).reshape(4, 1).astype(np.result_type(angles))
    _, prevInd = dfun.get_prev(angles[0, :])
    return angles[:, prevInd]


def new_star(label, Positions, AxisAngles, countData):
    # Initiates the information of a new star
    label += 1
    Positions[label] = [(None, None)] * (countData - 1)
    AxisAngles[label] = np.tile(np.vstack((None,) * 4), countData - 1)

    return label, Positions, AxisAngles


def actualize_star(particle, Positions, center, activeParticles, AxisAngles, currentAngles):
    # Actualizes the information of a given star
    Positions[particle].append(center)
    activeParticles.append(particle)
    AxisAngles[particle] = np.hstack((AxisAngles[particle], currentAngles))
    return Positions, activeParticles, AxisAngles


def double_label_check(label, trackedParticle, Positions, center, activeParticles, AxisAngles, currentAngles,
                       countData):
    cx, cy = center
    # Double label check
    prevPos = [cent for cent in Positions[trackedParticle] if cent != (None, None)]
    if len(prevPos) > 2:
        prevCenter = prevPos[-2]
        otherCenter = prevPos[-1]
        Dcurrent = np.linalg.norm((prevCenter[0] - cx, prevCenter[1] - cy))
        Dother = np.linalg.norm((prevCenter[0] - otherCenter[0], prevCenter[1] - otherCenter[1]))
        if Dcurrent < Dother:
            # Current is the real particle -> Delete the wrong one and give it a new label
            # Give the previous one a new label, but keeping "trackedParticle" to later actualize information
            label, Positions, AxisAngles = new_star(label, Positions, AxisAngles, countData)
            Positions, activeParticles, AxisAngles = actualize_star(label, Positions, otherCenter, activeParticles,
                                                                    AxisAngles, currentAngles)

            del Positions[trackedParticle][-1]
            AxisAngles[trackedParticle] = np.delete(AxisAngles[trackedParticle], -1, axis=1)

        else:
            # Current is a new particle
            label, Positions, AxisAngles = new_star(label, Positions, AxisAngles, countData)
            trackedParticle = label
    else:
        # We don't have enough information to decide, so we give the current one a new label
        label, Positions, AxisAngles = new_star(label, Positions, AxisAngles, countData)
        trackedParticle = label

    return label, trackedParticle, Positions, activeParticles, AxisAngles


def extend_angle_dict(AxisAngles):
    # Makes the transitions between quadrants continous
    period = 360
    for particle in AxisAngles:
        # We need to take out the "None" values
        Angles = AxisAngles[particle]
        for kk in range(4):
            FillInd = np.invert(np.equal(Angles[kk, :], None))  # Indices of points that are not "None"
            Angles[kk, FillInd] = extend_angle(Angles[kk, FillInd], period)
            AxisAngles[particle][kk, FillInd] = Angles[kk, FillInd]
    return AxisAngles


def extend_angle(Angles, period):
    # Makes the transitions between quadrants continuous
    prevAng = Angles[0]
    quadrant = np.floor(prevAng / period)
    for num in range(1, len(Angles)):
        currAng = Angles[num] + period * quadrant  # Move angle to current quadrant
        extendAng = currAng + period * np.array([-1, 0, 1])  # Extend to previous, current and next quadrant
        diffAng = extendAng - prevAng
        _, minInd = min((val, ind) for (ind, val) in
                        enumerate(np.abs(diffAng)))  # Get the quadrant corresponding to the closest angle
        quadrant += (
                    minInd - 1)  # 0 if no quadrant change, -1 if one quadrant to the left, +1 if one quadrant to the right
        currAng = Angles[num] + period * quadrant
        Angles[num] = prevAng = currAng
    return Angles


def fill_in_angle(AxisAngles, StartEnd, mode):
    # Interpolates missing angles marked with "None"
    newAxis = {}
    for particle in AxisAngles:
        startend = StartEnd[particle]
        angles = AxisAngles[particle]
        if None in list(angles[0, startend[0]:startend[1]]):
            newAxis[particle] = np.zeros((4, startend[1] - startend[0] + 1))
            for i in range(4):
                interAng = dfun.interpolate(angles[i, :], startend, mode)
                if interAng != 'wrong':
                    interAng = interAng.astype(int)
                    newAxis[particle][i, :] = interAng
        else:
            newAxis[particle] = AxisAngles[particle][:, startend[0]:startend[1] + 1]
    return newAxis


def get_angular_velocity(AxisAngles, order):
    # Returns dictionary with the instantaneous angular velocity calculated
    VelAng = {}
    for particle in AxisAngles:
        for i in range(4):
            angles = AxisAngles[particle][i]
            velang = dfun.centered_difference(angles, order)
        VelAng[particle] = velang
    return VelAng


def run_main(mode):
    global Vars, frame, debug
    # Mode must be "thresh" or "gradient"

    ##### INITIALISING PARAMETERS #####

    # # # GUI # # #
    trackInfo = ['position']
    root = tkinter.Tk()
    if mode == 'thresh':
        global detection_thresh
        import detection_thresh
        app = GUI_main.threshGUI(root)
    elif mode == 'gradient':
        global detection_gradient
        import detection_gradient
        app = GUI_main.gradientGUI(root, trackInfo)
    elif mode == 'absThresh':
        global detection_absThresh
        import detection_absThresh
        app = GUI_main.absThreshGUI(root)
    else:
        sys.exit('Wrong mode')

    root.mainloop()

    # Import variables in dictionary
    Vars = {}
    for var in app.variables:
        Vars[var] = app.variables[var].get()

    # For choosing the video
    vid = cv2.VideoCapture(Vars['filename'])
    width, height = int(vid.get(3)), int(vid.get(4))
    shape = (width, height)
    FPS = vid.get(5)

    try:
        vid_FPS = float(Vars['vid_FPS'])
        if vid_FPS == 0:
            raise ValueError
    except ValueError:
        vid_FPS = FPS
    vid_FPS /= float(Vars['vid_skip'])

    try:
        Vars['vid_end'] = float(Vars['vid_end'])
    except ValueError:
        Vars['vid_end'] = 0

    if Vars['vid_end'] < 1:
        Vars['vid_end'] = np.inf

    # For saving the video
    path_parts = Vars['filename'].split('/')
    path = '/'.join(path_parts[:-1]) + '/'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec
    out = cv2.VideoWriter(path + 'nanoDetection_' + path_parts[-1], fourcc, vid_FPS,
                          (width, height))  # Create VideoWriter object

    # Objects needed for tracking
    Positions = {}  # { particle : [(x0,y0),(x1,y1),...,(xf,yf)] }
    SavedPositions = {}
    Frames = []
    defunktParticles = set([])

    label = 0
    count = 0  # Counts frames
    countData = 0  # Counts analyzed frames (will vary from "count" when vid_skip > 1)
    avNum = 10  # Averages for vectorial velocity (takes 2*avNum+1 samples)
    boo, ff = dfun.needs_fitting(width, height, 1366, 768)
    needsFitting = (boo and Vars['vid_fit'])
    recording = True
    frameMode = False

    # Stars variables
    starRadi = Vars2['starRadi']
    ticks = Vars2['ticks']
    varray = get_rotating_vectors(ticks)  # Array with perpendicular rotating vectors
    AxisAngles = {}  # {particle : [angles1 angles2 ... anglesf]} with angles = [[axis1],[axis2]]

    StoreVars = Vars.copy()

    while (True):
        count += 1  # Number of frames
        ret, frame = vid.read()
        if ret == False:
            break

        # Skip frames
        if count % Vars['vid_skip'] != 0 or not Vars['vid_start'] <= count:
            continue

        # Stop video
        if not count <= Vars['vid_end']:
            break

        # Save frame number
        Frames.append(count)
        countData += 1

        original = frame.copy()

        # Treat image and return binary
        _, binary = binarization(frame, mode)
        # frame, binary = binarization(frame, mode)

        show = binary.copy()

        # Get rough contour from mask
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        activeParticles = []  # Will store particles in current frame

        ### SPECIAL STARS
        # Treat image again to eliminate noise
        for stringVar in additional_vars:
            # Treat the image differently to get the axis
            Vars[stringVar] = Vars2[stringVar]

        _, binary = binarization(frame, mode)
        Vars = StoreVars.copy()
        show2 = binary.copy()

        for cntInd in range(len(contours)):
            cnt = contours[cntInd]
            # Is it a nanomotor?
            judgement, M = particle_test(cnt)

            if judgement == True:

                # Calculate centroid and other information of the particle
                # Gotten from the non-denoised binary image
                area = M['m00']
                # cx, cy = np.around(np.average(cnt, axis = (0,1))).astype(int)
                cx = int(round(M['m10'] / area))
                cy = int(round(M['m01'] / area))
                center = np.array([cx, cy])

                # Mask the star contour for angle calculation using the denoised binary image
                maskStar = np.zeros((height, width), np.uint8)
                cv2.circle(maskStar, (cx, cy), starRadi, 255, -1)
                maskStar = cv2.bitwise_and(binary, maskStar)
                currentAngles, v1, ratioInfo = get_star_angles(maskStar, center, ticks, varray, starRadi)
                v2 = perpendicular(v1)

                # cv2.line(frame, tuple(center-v1), tuple(center+v1), (0,255,255),1)
                # cv2.line(frame, tuple(center-v2), tuple(center+v2), (0,255,255),1)

                #### Tracking (fast particles) ####
                # Searches in a circle around the particle

                found = False
                nearby = np.zeros((height, width), np.uint8)
                nearRadi = Vars['trk_radi']
                cv2.circle(nearby, (cx, cy), nearRadi, 255, -1)
                posibleParticles = []

                partIter = (prt for prt in Positions if prt not in defunktParticles)
                for particle in partIter:
                    # First check: location
                    prevCenter, _ = dfun.get_prev_c(Positions[particle])
                    location = (nearby[prevCenter[1], prevCenter[0]] == 255)
                    # cv2.circle(nearby,prevCenter,2,130,-1)

                    if location == True:
                        posibleParticles.append(particle)
                        found = True

                # Tracked particle is the label corresponding to the current contour
                if found == False:
                    # If it's not inside the contour, then it is a new particle
                    label, Positions, AxisAngles = new_star(label, Positions, AxisAngles, countData)
                    trackedParticle = label

                else:
                    if len(posibleParticles) == 1:
                        # Only one possibility
                        trackedParticle = posibleParticles[0]

                    elif len(posibleParticles) > 1:
                        # More than one posibility: we choose the closest one (Bayesian decision making)
                        dd = [np.linalg.norm(
                            (dfun.get_prev_c(Positions[i])[0][0] - cx, dfun.get_prev_c(Positions[i])[0][1] - cy)) for i
                              in posibleParticles]
                        trackedParticle = posibleParticles[dd.index(min(dd))]

                    previousAngles = get_prev_ang(AxisAngles[trackedParticle]).reshape(4, 1)
                    currentAngles = track_angle(previousAngles, currentAngles)

                # Actualize relevant information

                if trackedParticle in activeParticles:
                    # Double label check
                    label, trackedParticle, Positions, activeParticles, AxisAngles = double_label_check(label,
                                                                                                        trackedParticle,
                                                                                                        Positions,
                                                                                                        center,
                                                                                                        activeParticles,
                                                                                                        AxisAngles,
                                                                                                        currentAngles,
                                                                                                        countData)

                Positions, activeParticles, AxisAngles = actualize_star(trackedParticle, Positions, (cx, cy),
                                                                        activeParticles, AxisAngles, currentAngles)

                # # Draw label
                pt = (cx + 25, cy + 25)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(frame,str(trackedParticle),tuple(pt),font,0.5,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(frame, str(ratioInfo[2])[:5], tuple(pt), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

                # Draw angle and follow axis 1
                # angle = AxisAngles[trackedParticle][0,-1]
                # vector = (starRadi * vec_from_angle(angle)).astype(int)[::-1]
                # cv2.putText(frame, str(1), tuple(Positions[trackedParticle][-1]+vector), font,0.5,(0,255,0),2,cv2.LINE_AA)

        # Check what particles are not on frame and mark the old ones
        for particle in list(Positions.keys()):
            isthere = particle in activeParticles
            if not isthere:
                Positions[particle].append((None, None))
                AxisAngles[particle] = np.hstack((AxisAngles[particle], np.vstack((None,) * 4)))

            # Mark old particles
            if particle not in defunktParticles:
                lastSeen = -dfun.get_prev_c(Positions[particle])[1]
                if lastSeen > Vars['trk_memory']:
                    defunktParticles.add(particle)

        # Show counter
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(count), (5, 15), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Fitting in 1366x768 monitor
        if needsFitting == True:
            notFittedFrame = frame.copy()
            frame_show = cv2.resize(frame, None, fx=ff, fy=ff, interpolation=cv2.INTER_LINEAR)
            show = cv2.resize(show, None, fx=ff, fy=ff, interpolation=cv2.INTER_LINEAR)
        else:
            frame_show = frame.copy()

        if Vars['vid_contour']:
            cv2.imshow('contour', show)
            cv2.imshow('denoised', show2)

        # frame_show = color2gray(frame_show,1)
        cv2.imshow('show', frame_show)
        # cv2.imshow('frame',original)

        ch = cv2.waitKey(1) & 0xFF

        if ch == 27:
            # Exit
            break
        elif ch == ord('z'):
            # Save frame
            cv2.imwrite(path + 'frame.png', original)
        elif ch == ord('p') or frameMode == True:
            # Pause
            ch2 = cv2.waitKey() & 0xFF
            if ch2 == 81 or ch2 == 83:
                # Next Frame
                frameMode = True
            else:
                frameMode = False
        elif ch == ord('r'):
            # Toggle resize
            needsFitting = not needsFitting

        if recording == True:
            out.write(frame)

    out.release()
    vid.release()
    cv2.destroyAllWindows()

    AxisSave = copy.deepcopy(AxisAngles)

    #### Choose particles to analyze ####

    root = tkinter.Tk()
    app2 = GUI_main.dataAnalysisGUI(root)
    root.mainloop()

    # Import variables
    for var in app2.variables:
        Vars[var] = app2.variables[var].get()

    print('Analyzing...')

    # Time units
    try:
        data_FPS = float(Vars['data_FPS'])
        unit_time = 'seconds'
        if data_FPS == 0.:
            raise ValueError
    except ValueError:
        data_FPS = 1
        unit_time = 'frames'
    Frames = list(np.array(Frames) / data_FPS)

    # Distance units
    try:
        data_pix2mtr = float(Vars['data_pix2mtr'])
        unit_distance = 'microm'
        if data_pix2mtr == 0.:
            raise ValueError
    except ValueError:
        data_pix2mtr = 1
        unit_distance = 'pixels'

    #### Purge the particles that have not been selected ####

    if len(Vars['data_ParticleList']) >= 1:
        data_ParticleList = [int(num) for num in Vars['data_ParticleList'].split(',')]
        for particle in list(Positions.keys()):
            if particle not in data_ParticleList:
                del Positions[particle]
                del AxisAngles[particle]

    StartEnd = dfun.get_startend(Positions)

    #### Saving data ####
    dfun.save_data_raw(path + 'data_raw_' + path_parts[-1][:-4] + '.csv', Positions, Frames, unit_distance, unit_time)

    #### Interpolate missing points for automatic MSD ####
    interpMode = Vars['data_interp']
    Positions = dfun.fill_in(Positions, StartEnd, interpMode)
    AxisAngles = extend_angle_dict(AxisAngles)  # Make the angle transitions between quadrants continuous
    AxisAngles = fill_in_angle(AxisAngles, StartEnd, interpMode)  # Interpolate
    if Vars['data_draw'] == True:
        drawPositions = Positions.copy()

    # Convert units
    for particle in list(Positions.keys()):
        pos = Positions[particle]
        Positions[particle] = list(zip(*list(zip(*(np.array(pos) / data_pix2mtr)))))

    #### Centered derivative to get velocity ####

    VelVec, VelMod, VelTotal = dfun.get_velocity(Positions, Vars['data_order'])
    VelAng = get_angular_velocity(AxisAngles, Vars['data_order'])

    #### Saving data ####
    with open(path + 'data_inter_' + path_parts[-1][:-4] + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for particle in Positions:
            X, Y = list(zip(*Positions[particle]))
            start, end = StartEnd[particle]
            writer.writerow(['Particle', str(particle)])
            indexes = (start, end + 1)
            timeUnits = Frames[indexes[0]:indexes[1]]
            writer.writerow(['Time (' + unit_time + ')'] + timeUnits)
            writer.writerow(['X (' + unit_distance + ')'] + list(X))
            writer.writerow(['Y (' + unit_distance + ')'] + list(Y))
            writer.writerow(['Angle (degrees)'] + list(AxisSave[particle][0, start:end + 1]))
            writer.writerow(['Angle 2 (degrees)'] + list(AxisSave[particle][1, start:end + 1]))
            writer.writerow(['Angle 3 (degrees)'] + list(AxisSave[particle][2, start:end + 1]))
            writer.writerow(['Angle 4 (degrees)'] + list(AxisSave[particle][3, start:end + 1]))
            writer.writerow(['Angle (degrees)'] + list(AxisAngles[particle][0, :]))
            writer.writerow(['Angle 2 (degrees)'] + list(AxisAngles[particle][1, :]))
            writer.writerow(['Angle 3 (degrees)'] + list(AxisAngles[particle][2, :]))
            writer.writerow(['Angle 4 (degrees)'] + list(AxisAngles[particle][3, :]))
            writer.writerow(['Angular velocity (degrees/' + unit_time + ')'] + list(VelAng[particle]))
            incT, MSD = dfun.mean_square_displacement(X, Y)
            # incT = [i/float(FPS) for i in incT]
            writer.writerow(['Time displacement (' + unit_time + ')'] + list(Vars['vid_skip'] * incT / data_FPS))
            writer.writerow(['MSD (' + unit_distance + '**2)'] + list(MSD))
            # Angular mean square displacement
            angles = AxisAngles[particle][0, :]
            _, AngDisp = dfun.mean_square(angles)
            writer.writerow(['Angle square displacement (' + unit_time + ')'] + list(AngDisp))

            mod = data_FPS * VelMod[particle] / Vars['vid_skip']
            writer.writerow(['Velocity Modulus (' + unit_distance + '/' + unit_time + ')'] + list(mod))
            modAv = np.average(mod)
            writer.writerow(['Average vel modulus', str(modAv)])
            velTot = VelTotal[particle]
            velTot = velTot * (data_FPS / Vars['vid_skip'])
            writer.writerow(['Total distance / Total time (' + unit_distance + '/' + unit_time + ')', str(velTot)])
            writer.writerow('')

        sorted_keys = sorted(Vars.keys())
        sorted_values = [Vars[key] for key in sorted_keys]
        writer.writerow(['Parameter name:'] + sorted_keys)
        writer.writerow(['Parameter value:'] + sorted_values)
        writer.writerow('')

        sorted_keys = sorted(Vars2.keys())
        sorted_values = [Vars2[key] for key in sorted_keys]
        writer.writerow(['Parameter name:'] + sorted_keys)
        writer.writerow(['Parameter value:'] + sorted_values)
        writer.writerow('')

    print('Done')

    #### Draw trajectories ####
    if Vars['data_draw'] == True:

        data_color = [Vars['data_colorB'], Vars['data_colorG'], Vars['data_colorR']]
        for i in range(3):
            if data_color[i] == 0:
                data_color[i] = 1

        # For choosing the video
        vid = cv2.VideoCapture(Vars['filename'])
        width, height = int(vid.get(3)), int(vid.get(4))
        shape = (width, height)
        FPS = vid.get(5)

        # For saving the video
        path_parts = Vars['filename'].split('/')
        path = '/'.join(path_parts[:-1]) + '/'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec
        out = cv2.VideoWriter(path + 'nanoDetection_marker_' + path_parts[-1], fourcc, vid_FPS,
                              (width, height))  # Create VideoWriter object

        # Variables
        prev_count = countData  # Previous data count
        count = 0  # Counts frames
        countData = 0  # Counts analyzed frames (will vary from "count" when vid_skip > 1)
        recording = True
        frameMode = False
        imgTail = np.zeros((height, width, 3),
                           dtype=np.uint8)  # Blank image used to save the trajectories before overlapping to the current frame

        # Position iterators
        PosIter = {}
        for particle in Positions:
            # The iterator will go through position indexes
            PosIter[particle] = iter(list(range(len(drawPositions[particle]))))

        while (True):
            count += 1  # Number of frames
            ret, frame = vid.read()
            if ret == False:
                break

            # Skip frames
            if count % Vars['vid_skip'] != 0 or not (Vars['vid_start'] <= count <= Vars['vid_end']):
                continue
            elif (countData + 1) == prev_count:
                break
            # Save frame number
            countData += 1
            for particle in drawPositions:

                start, end = StartEnd[particle]
                if start <= countData - 1 < end:
                    posIndex = next(PosIter[particle])
                    if start == countData - 1:
                        point1 = point2 = drawPositions[particle][posIndex]
                    else:
                        point1 = drawPositions[particle][posIndex - 1]
                        point2 = drawPositions[particle][posIndex]

                    cv2.line(imgTail, point1, point2, data_color, 1)

                    # Draw label
                    pt = point2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(frame,str(particle),(pt[0]+15,pt[1]+15),font,0.5,(0,0,255),2,cv2.LINE_AA)
                    cv2.circle(frame, tuple(pt), 1, (0, 255, 0), -1)

                    # Draw axis
                    angles = AxisAngles[particle][:, posIndex]
                    for num, ang in enumerate(angles):
                        vec = vec_from_angle(ang % 360)[::-1]
                        pt2 = np.array(pt) + Vars2['starRadi'] * vec
                        pt2 = np.around(pt2).astype(int)
                        if num == 0:
                            cv2.line(frame, tuple(pt), tuple(pt2), (0, 0, 255), 1)
                        else:
                            cv2.line(frame, tuple(pt), tuple(pt2), (0, 255, 255), 1)

            frame = np.where(imgTail, imgTail, frame)

            # Show counter
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(count), (5, 15), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

            # Fitting in 1366x768 monitor
            if needsFitting == True:
                notFittedFrame = frame.copy()
                frame_show = cv2.resize(frame, None, fx=ff, fy=ff, interpolation=cv2.INTER_LINEAR)
                show = cv2.resize(show, None, fx=ff, fy=ff, interpolation=cv2.INTER_LINEAR)
            else:
                frame_show = frame.copy()

            cv2.imshow('frame', frame)

            ch = cv2.waitKey(1) & 0xFF

            if ch == 27:
                # Exit
                break
            elif ch == ord('p') or frameMode == True:
                # Pause
                ch2 = cv2.waitKey() & 0xFF
                if ch2 == 81 or ch2 == 83:
                    # Next Frame
                    frameMode = True
                else:
                    frameMode = False

            if recording == True:
                out.write(frame)

        out.release()
        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # root = Tkinter.Tk()
    # chooseMethod = GUI_main.chooseMethod(root)
    # root.mainloop()
    # run_main(chooseMethod.method.get())
    run_main('gradient')

    # def analyze_frame(self):
    #     #Special stars
    #     import detection_main as det_main
    #     import detection_functions as dfun
    #     import detection_stars as stars

    #     additional_vars = ['starRadi', 'ticks', 'img_open', 'img_close', 'sobel_thresh', 'img_gaussblur', 'img_medianblur', 'img_bilateral_d', 'img_bilateral_color', 'img_bilateral_space', 'power']

    #     Vars2 = {}
    #     Vars2['starRadi'] = 20
    #     Vars2['ticks'] = 15
    #     Vars2['img_open'] = 1
    #     Vars2['img_close'] = 1
    #     Vars2['sobel_thresh'] = 190
    #     Vars2['img_gaussblur'] = 1
    #     Vars2['img_medianblur'] = 1
    #     Vars2['img_bilateral_d'] = 8
    #     Vars2['img_bilateral_color'] = 75
    #     Vars2['img_bilateral_space'] = 75
    #     Vars2['power'] = 8.0

    #     #Works with the global "Vars" dictionary
    #     self.Vars = {}
    #     for var in self.variables:
    #         self.Vars[var] = self.variables[var].get()

    #     det_main.Vars = self.Vars
    #     label = 0

    #     starRadi = Vars2['starRadi']
    #     ticks = Vars2['starRadi']
    #     varray = stars.get_rotating_vectors(ticks)
    #     self.StoreVars = self.Vars.copy()

    #     frame = self.catch_frame()
    #     original = frame.copy()
    #     self.fheight, self.fwidth = frame.shape[:2]

    #     #print det_main.Vars['img_bilateral_d']
    #     frame, binary = det_main.binarization(frame, self.mode)
    #     binary_show = binary.copy()

    #     frame_contour, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #     ### SPECIAL STARS
    #     #Treat image again to eliminate noise
    #     for stringVar in additional_vars:
    #         #Treat the image differently to get the axis
    #         self.Vars[stringVar] = Vars2[stringVar]

    #     det_main.Vars = self.Vars
    #     #print det_main.Vars['img_bilateral_d']
    #     _, binary =  det_main.binarization(original, self.mode)
    #     self.Vars = self.StoreVars.copy()
    #     binary_show = binary.copy()

    #     for cnt in contours:
    #         judgement, M = det_main.particle_test(cnt)
    #         if judgement == True:
    #             label += 1
    #             area = M['m00']
    #             cx = int(round(M['m10']/area))
    #             cy = int(round(M['m01']/area))
    #             center = np.array([cx,cy])

    #             maskStar = np.zeros((self.fheight,self.fwidth), np.uint8)
    #             cv2.circle(maskStar, (cx,cy), starRadi, 255, -1)
    #             maskStar = cv2.bitwise_and(binary,maskStar)
    #             stars.frame = frame
    #             currentAngles, v1, ratioInfo = stars.get_star_angles(maskStar, center, ticks, varray, starRadi)
    #             frame = stars.frame

    #             # Draw label
    #             pt = (cx+15,cy+15)
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             #cv2.putText(frame,str(trackedParticle),tuple(pt),font,0.5,(0,0,255),2,cv2.LINE_AA)
    #             cv2.putText(frame,str(ratioInfo[2])[:5],tuple(pt),font,0.5,(0,0,255),2,cv2.LINE_AA)
    #             cv2.circle(frame,(cx,cy),1,(0,255,0),-1)

    #     #Write frame
    #     cv2.putText(frame,str(self.count),(5,15),self.font,0.5,(0,0,255),2,cv2.LINE_AA)

    #     #Resize image based on monitor resolution
    #     temproot = Tkinter.Tk()
    #     self.swidth = temproot.winfo_screenwidth()
    #     self.sheight = temproot.winfo_screenheight()
    #     temproot.destroy()

    #     pheight, pwidth = binary.shape

    #     #The image needs to fit verticaly in a little less than half of the screen
    #     needsFitting, ff = dfun.needs_fitting(pwidth, pheight, 0.9*self.swidth, 0.5*self.sheight-25) 

    #     if needsFitting:
    #         frame = cv2.resize(frame,None,fx = ff, fy = ff, interpolation = cv2.INTER_LINEAR)
    #         binary_show = cv2.resize(binary_show,None,fx = ff, fy = ff, interpolation = cv2.INTER_LINEAR)

    #     #Convert result to img type readable by Tkinter
    #     PILframe = Image.fromarray(frame)
    #     tkframe = ImageTk.PhotoImage(PILframe)

    #     PILbinary = Image.fromarray(binary_show)
    #     tkbinary = ImageTk.PhotoImage(PILbinary)
    #     return tkframe, tkbinary

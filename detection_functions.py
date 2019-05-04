import random  # For random numbers and selections
import os  # For directory finding
import sys
import copy  # For deepcopy
import csv  # For excel writing
import numpy as np  # Arrays
import matplotlib.pyplot as plt  # Plotting
import time  # Calculate function
from scipy import ndimage  # Image processing
from scipy.optimize import curve_fit  # Curve fitting
from scipy.interpolate import interp1d
from scipy.stats import linregress
import cv2  # OpenCV
import tkinter  # GUI interface
import GUI_main  # GUI classes
import math


# [modified]
# add a function convert HSV color to RGB
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return b, g, r


# add a function covert speed to color (blue to red)
def speed2hsv(speed, min, max):
    return (max - speed) / (max - min) * 240, 1, 1


# [modified end]

##### GENERAL USEFUL FUNCTIONS #####

def circular_kernel(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x <= r * r).astype(np.uint8)


def average(vector):
    # Averages a vector, does not crash if it's a number or an empty list
    if isinstance(vector, (int, float)):
        return vector
    else:
        length = len(list(vector))
        if length != 0:
            return int(round(sum(vector) / float(length)))
        else:
            return 0


def get_prev_c(data, num=1):
    # Find the "num" center (counting from the back) in list "data" that is not "None".
    counter = 1
    found_counter = 0
    l = len(data)
    while counter <= l:
        previous = data[-counter]
        if previous != (None, None):
            found_counter += 1
            if found_counter == num:
                return previous, -counter
        counter += 1
    #    print 'damn'
    #    print data
    return (None, None), -counter


def get_prev_consecutive_c(data, num=1):
    # Returns a total of consecutive "num" centers (counting from the back). Stops if it finds a "None"
    counter = 1;
    found_counter = 0
    c_list = []
    while counter <= len(data):
        previous = data[-counter]
        if previous != (None, None):
            c_list.append(previous)
            found_counter += 1
            if found_counter == num:
                break
        else:
            break
    return c_list


def get_prev(data):
    # Find the last value in list "data" that is not "None"
    counter = 1
    while counter <= len(data):
        previous = data[-counter]
        if previous != None:
            return previous, -counter
        counter += 1
    return None, -counter


def bound(shape, point):
    # Checks if a point is out of bounds of an array. If so returns a bounded point
    w, h = shape
    point = np.around(point).astype(int)

    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

    if point[0] >= w and point[1] >= h:
        return np.array([w - 1, h - 1])
    elif point[0] >= w:
        return np.array([w - 1, point[1]])
    elif point[1] >= h:
        return np.array([point[0], h - 1])
    else:
        return point


def my_index(l, a):
    # Finds the index of the first instance of "a" in list "l". Return length of list if not found
    try:
        index = l.index(a)
    except ValueError:
        index = len(l)
    return index


def store(container, item):
    # Pushes item in a container (numbered dict) of fixed size.
    # Beispiel: {0:104, 1:103, 2:102, 3:101}, 105 => {0:105, 1:104, 2:103, 3:102}
    return [item] + container[:-1]


def limit(array, low_limit, high_limit):
    # Returns all the values in "array" in the range defined by the low and high limits
    return np.minimum(np.maximum(array, low_limit), high_limit)


def get_closest(seq, num):
    # Gets the value in seq closest to num from the bottom (Input: [0, 10, 20], 18 Output: 10)
    return seq[sum((num - np.array(seq)) >= 0) - 1]


def monitor_resolution():
    # Calculate monitor resolution
    temproot = tkinter.Tk()
    swidth = temproot.winfo_screenwidth()
    sheight = temproot.winfo_screenheight()
    temproot.destroy()
    return (swidth, sheight)


class warped_iter():

    # Simple iterator class that starts again once it reaches the end

    def __init__(self, original_list):
        self.original = original_list
        self.create_iter()

    def iter_next(self):
        try:
            val = next(self.iterator)

        except StopIteration:
            self.create_iter()
            val = next(self.iterator)

        return val

    def create_iter(self):
        self.iterator = iter(self.original)


def interpret_list(data_string):
    # Input list is a string of elements. Individual elements are separated by commas, consecutive elements are marked with double dot.
    data_list = []
    for num in data_string.split(','):
        if ':' in num:
            num1, num2 = num.split(':')
            if num1 == '' or num2 == '':
                nums = [num1, num2]
                data_list.append(nums[nums[0] == ''])  # Append the correct number, ignore the other
            else:
                data_list.extend(list(range(int(num1), int(num2) + 1)))
        elif num == '':
            continue  # Empty string
        else:
            data_list.append(int(num))
    return data_list


def get_points_in_cnt(cnt):
    # Returns points inside contour in the form of [[p0x, p0y],[p1x,p1y],...] (coordinates use array system of reference)
    x, y, w, h = cv2.boundingRect(cnt)  # For efficiency purposes we get the area around the contour
    cnt = cnt - np.array([x, y])  # Displaced contour
    p_list = []
    for i in range(w):
        for j in range(h):
            p = (i, j)
            if cv2.pointPolygonTest(cnt, p, False) > 0:
                p_list.append(p)  # Point is inside

    p_list = [[py + y, px + x] for (px, py) in p_list]
    return p_list


##### DRAWING FUNCTIONS #####


def draw_arrow(img, point1, point2, armAngle, color, thickness):
    # Draw a fucking arrow
    # armAngle in radians
    point1 = np.array(point1);
    point2 = np.array(point2)
    vector = point2 - point1
    cv2.line(img, tuple(point1), tuple(point2), color, thickness)
    dist = np.linalg.norm(vector)
    angle = np.arctan2(vector[1], vector[0])
    angle = angle % (2 * np.pi)  # map from (-180,180) to (0,360)
    angle1 = np.pi + angle + armAngle
    angle2 = np.pi + angle - armAngle
    arm1 = (dist / 4) * np.array([np.cos(angle1), np.sin(angle1)])
    arm2 = (dist / 4) * np.array([np.cos(angle2), np.sin(angle2)])
    cv2.line(img, tuple(map(int, point2)), tuple(map(int, point2 + arm1)), color, thickness)
    cv2.line(img, tuple(map(int, point2)), tuple(map(int, point2 + arm2)), color, thickness)


def chunks(l, n):
    # Generator that returns consecutive pieces of the "l" list of size "n"
    for i in range(0, len(l), n):
        yield l[i:(i + n)]


def dashed_line(img, point1, point2, dashSize, color, thickness):
    # Draws a dashed line. Dashes are "dashSize" pixel in length
    point1 = np.array(point1);
    point2 = np.array(point2)
    x, y = get_line_points(point1, point2)
    xiter, yiter = chunks(x, dashSize), chunks(y, dashSize)
    while True:
        try:
            img[next(yiter), next(xiter)] = color  # Paint dash
            next(xiter);
            next(yiter)  # Leave dash blank

        except StopIteration:
            break
    return img


def draw_cross(img, point, s, color):
    # Draws a cross with arms of size "s"

    cv2.line(img, tuple(point), (point[0] + s, point[1]), color, 1)
    cv2.line(img, tuple(point), (point[0], point[1] + s), color, 1)
    cv2.line(img, tuple(point), (point[0] - s, point[1]), color, 1)
    cv2.line(img, tuple(point), (point[0], point[1] - s), color, 1)


def get_line_points(start, end):
    # Return all the points in a line going from start to end
    dist = int(round(np.linalg.norm(end - start)))
    x = np.linspace(start[0], end[0], dist)
    y = np.linspace(start[1], end[1], dist)
    x, y = np.around(x).astype(int), np.around(y).astype(int)
    return [x, y]


def bresenham(start, end):
    # Returns all the points in a line going from start to end using Bresenham's algorithm
    linePoints = [[], []]  # Will store points [[x0,x1,x2,...],[y0,y1,y2]] for fancy indexing
    y0, x0 = start;
    y1, x1 = end
    if x0 != x1 and y0 != y1:
        # The code is made for x0 < x1
        # The line is better when we iterate over the axis with the most points
        deltax = x1 - x0;
        deltay = y1 - y0

        flipped = False
        if abs(deltax) < abs(deltay):
            # Y axis has more points than x -> Flip
            x0, y0, x1, y1 = y0, x0, y1, x1
            deltax, deltay = deltay, deltax
            flipped = True  # Remember to revert afterwards!

        if x0 > x1:
            # start is to the left of end, invert start/end points
            x0, y0, x1, y1 = x1, y1, x0, y0
            deltay = -deltay

        error = 0
        deltaerror = abs(deltay / float(deltax))
        y = y0

        for x in range(x0, x1 + 1):

            if not flipped:
                linePoints[0].append(x);
                linePoints[1].append(y)
            else:
                linePoints[0].append(y);
                linePoints[1].append(x)

            error += deltaerror
            while error >= 0.5:
                y += np.sign(deltay)
                error -= 1

    elif x0 == x1:
        y0, y1 = np.sort([y0, y1])  # From smallest to largest
        points = list(range(y0, y1 + 1))
        linePoints = [[x0] * len(points), points]
    elif y0 == y1:
        x0, x1 = np.sort([x0, x1])
        points = list(range(x0, x1 + 1))
        linePoints = [points, [y0] * len(points)]
    return linePoints


def get_vector_circle(radi=64, ticks=180):
    # Returns a set of vectors of radi "radi" between 0 and 180 degrees
    if type(ticks) == str:
        if ticks == 'MAX':
            ticks = np.clip(radi ** 3, 0, 5000)
            ticks = 63
        else:
            ticks = int(ticks)

    ang = np.linspace(0, np.pi, ticks + 1)[:-1]
    pos = radi * np.vstack((np.cos(ang), np.sin(ang))).T
    pos = np.around(pos).astype(int)
    pos = np.array(list({tuple(i) for i in pos}))
    ang = np.around(np.rad2deg(np.arctan2(pos[:, 1], pos[:, 0]))).astype(int)
    vecDict = {ang: pos[ind, :] for ind, ang in enumerate(ang)}
    return vecDict


def get_max_ticks(radi):
    # Returns the maximum amount of angles that you can possibly get with this radi
    ticks = radi ** 3
    if radi > 100:
        ticks = 5000
    maxnum = len(get_vector_circle(radi=radi, ticks=ticks))
    return maxnum


def get_colors(number):
    # Returns a list of equally spaced colors with length 'num'
    huelimit = 180
    color_bin = huelimit / number  # Hue is in degrees (0 to 360)
    colorList = []  # np.zeros((number,1,3))

    for i in range(number):
        # Get random values in HSV color space
        # Hue defines the color. The colors are random, but always within specified bins so that they are not repeated.
        hue = (2 * i + 1) * color_bin / 2  # Color centered in its respective bin
        sat = 220  # We want relatively high saturation
        val = 240  # And relatively high value
        hsv = np.array([hue, sat, val])  # np.array([limit(hue, 0, huelimit), limit(sat, 0, 255), limit(val, 0, 255)])

        hsv = hsv.astype(np.uint8).reshape((1, 1, 3))  # Limit to 0-255 (no overflow)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Transform to RGB
        # [modified] change color type to int from np.int32
        bgr_int = bgr.ravel().astype(int)
        bgr_int = [int(bgr_int[0]), int(bgr_int[1]), int(bgr_int[2])]
        colorList.append(list(bgr_int))

    # [modified] error in py3, slice index should be integer
    # start = number / 3 - 1
    start = number // 3 - 1
    colorList = colorList[start:] + colorList[:start]
    return colorList


def get_rand_colors(number):
    # Returns a list of random colors with length 'num' (equally separated in the HSV color space)
    huelimit = 180
    color_bin = huelimit / number  # Hue is in degrees (0 to 360)
    colorList = []  # np.zeros((number,1,3))

    for i in range(number):
        # Get random values in HSV color space
        # Hue defines the color. The colors are random, but always within specified bins so that they are not repeated.
        hue = np.random.normal((2 * i + 1) * color_bin / 2, color_bin / 4)  # Color centered in its respective bin
        sat = np.random.normal(220, 35)  # We want relatively high saturation
        val = np.random.normal(240, 15)  # And relatively high value
        hsv = np.array([limit(hue, 0, huelimit), limit(sat, 0, 255), limit(val, 0, 255)])

        hsv = hsv.astype(np.uint8).reshape((1, 1, 3))  # Limit to 0-255 (no overflow)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Transform to RGB
        colorList.append(list(bgr.ravel().astype(int)))

    random.shuffle(colorList)
    return colorList


##### IMAGE TREATMENT FUNCTIONS #####

def smoothing(image, Vars):
    # Transform to grayscale
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Treat the image to reduce noise
    # Smoothing
    if Vars['img_bilateral_d'] != 0:
        frame_gray = cv2.bilateralFilter(frame_gray, Vars['img_bilateral_d'], Vars['img_bilateral_color'],
                                         Vars['img_bilateral_space'])
    frame_gray = cv2.medianBlur(frame_gray, Vars['img_medianblur'])  # Good for eliminating white noise
    frame_gray = cv2.GaussianBlur(frame_gray, (Vars['img_gaussblur'],) * 2, 0)  # Good for general smoothing
    frame = frame_gray.copy()
    frame = gray2color(frame)

    return frame, frame_gray


def morphology(image, Vars):
    # Applies different morphological operations to the image, which should be binary

    kernel = np.ones((Vars['img_close'],) * 2, np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Eliminate small errors
    kernel = np.ones((Vars['img_open'],) * 2, np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # Eliminate small errors

    return image


def color_invert(color, ref=255):
    # Inverts color pixel in RGB or GBR. Can do less than half a loop by changing "ref"
    # [modified] change color scalar type to int from np.int32
    w = np.uint8(ref)
    c = w - np.array(color)
    c = [int(c[0]), int(c[1]), int(c[2])]
    return c
    # return w-np.array(color)


def color_fix(color):
    # Cheap hack so that it's possible to use the "np.where" function when overlapping (substitutes 0 by 1)
    # [modified] error in py3(or opencv), change 'np.int32' to 'int'
    # fixedColor = color
    fixedColor = [int(color[0]), int(color[1]), int(color[2])]
    for i in range(3):
        if color[i] == 0:
            color[i] = 1
    return fixedColor


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey() & 0xFF
    cv2.destroyAllWindows()


def gray2color(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def color2gray(img, mode):
    # Converts color to gray. Two methods (1 is recommended)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if mode == 1:
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    elif mode == 2:
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    gray = np.around(gray)
    gray = gray.astype(np.uint8)
    return gray


def needs_fitting(width, height, swidth, sheight):
    # For showing the video in a monitor with swidth x sheight size

    if width > swidth or height > sheight:
        needsFitting = True
        scale_x = 1;
        scale_y = 1
        if width > swidth:
            scale_x = 0.9 * swidth / width
        if height > sheight:
            scale_y = 0.85 * sheight / height
        ff = min(scale_x, scale_y)
    else:
        needsFitting = False
        ff = 1
    return needsFitting, ff


def draw_line(img, point1, point2, color, thick):
    # Equivalent to cv2 "line" function, but returns the painted points
    global dilation_ref
    try:
        ref
    except NameError:
        ref = dilation_ref(thick)
    linepoints = get_line_points(np.array(point1[::-1]), np.array(point2[::-1]))
    linepoints = dilation(linepoints, thick, ref)
    img[list(zip(*linepoints))] = color
    return img, linepoints


def dilation_ref(th):
    # Returns reference for point dilation with thickness "th"
    ref = np.zeros((2 * th + 1,) * 2)
    cv2.circle(ref, (th,) * 2, th, 1, -1)  # Reference circle of dilation
    ref_points = np.where(ref == 1)  # Coordinates of dilated reference points
    return ref_points


def dilation(linepoints, th, ref):
    # Dilates a list of points in the form [[cx0, cx1, ...], [cy0, cy1, ...]] by the specified thickness
    if th == 1:
        return linepoints

    points = list(zip(linepoints[0], linepoints[1]))
    points_dil = [[], []]
    points_set = set([])

    for pp in points:
        points_dil[0].extend(ref[0] + pp[0] - th)
        points_dil[1].extend(ref[1] + pp[1] - th)
    points_set.update(list(zip(points_dil[0], points_dil[1])))  # Eliminate equal points
    return points_set


def split_particles_ROI(radi, cnt):
    # Prepares the ROI for particle splitting
    x, y, w, h = cv2.boundingRect(cnt)  # For efficiency purposes we get the area around the contour
    ROI_touching = np.zeros((h, w), dtype=np.uint8)
    cnt = cnt - np.array([x, y])
    cv2.drawContours(ROI_touching, [cnt], 0, 255, -1)

    #### SCALE ####
    ff = 3
    ROI_touching = cv2.resize(ROI_touching, None, fx=ff, fy=ff, interpolation=cv2.INTER_LINEAR)
    radi = ff * radi
    #### SCALE ####

    cnt_list_split = split_particles(ROI_touching, radi)
    cnt_list_split = [c / ff + np.array([x, y]) for c in
                      cnt_list_split]  # Moving the coordinates from ROI to real image
    return cnt_list_split


def split_particles(binary, radi):
    # Splits touching, spherical particles with erosion. Returns contours of individual particles.
    r = int(radi * 0.75)  # Radius of the eroding kernel
    kernel = circular_kernel(r)
    #
    #    #Method 1: contour is gotten from watershed (accurate borders, computationally expensive)
    #    eroded = cv2.erode(binary, kernel, iterations =1) #Erosion gets the individual centers of the particles
    #
    #    ret, markers = cv2.connectedComponents(eroded) #Components are labeled 1,2,3,...
    #    markers = markers + 1 #We do this to make sure background is considered 1
    #    unknown = binary-eroded #Unknown area
    #    markers[unknown == 255] = 0
    #    markers = cv2.watershed(gray2color(binary), markers) #Watershed to separate components
    #    #original[eroded == 255] = [255,0,0]
    #    #print 'hey'
    #    showw = np.uint8(markers * (255.0/np.max(markers)))
    #    showw[eroded == 255] = 0
    ##    cv2.imshow('hjg',showw)
    ##    cv2.waitKey(1) & 0xFF
    #
    #    cnt_list_split = []
    #    for label in range(2,np.max(markers)+1):
    #        split_particle = np.zeros(binary.shape, np.uint8)
    #        split_particle[markers == label] = 255
    #        _,split_cnt,_ = cv2.findContours(split_particle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #        #cv2.drawContours(original, split_cnt, -1, [0,255,0], 2)
    #
    #        cnt_list_split.append(split_cnt[0])
    #
    #    return cnt_list_split
    ##
    # Method 2: contour is gotten from dilation (resulting borders overlap and are not very accurate, but it's fast)
    eroded = cv2.erode(binary, kernel, iterations=1)  # Erosion gets the individual centers of the particles
    erode_contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt_list_split = []  # List with
    for cnt in erode_contours:
        split_particle = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(split_particle, [cnt], 0, 255, -1)
        # split_particle = cv2.morphologyEx(split_particle, cv2.MORPH_OPEN, np.ones((9,9)))
        split_particle = cv2.dilate(split_particle, kernel, iterations=1)
        split_cnt, _ = cv2.findContours(split_particle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #        cv2.drawContours(original, split_cnt, -1, [0,255,0], 1)

        cnt_list_split.append(split_cnt[0])

    return cnt_list_split


def otsu_thresh(grayvals):
    # Performs otsu's algorithm to a list of grayvalues. Returns the threshold that minimizes intra-class variance
    grayvals = sorted(grayvals)
    hist, _ = np.histogram(grayvals, bins=255, range=(0, 255), density=True)
    t_low = grayvals[0]
    t_high = grayvals[-1]
    ind = 0
    std_inter = []
    w0 = 0
    w1 = 1
    for t in range(t_low, t_high):  # Advance threshold value
        n = grayvals[ind]
        if n == t:  # If the next values are not above the threshold it doesn't make sense
            n2 = n
            while n == n2:  # Searches for the last time a gray value equal to the threshold appears
                ind += 1
                n2 = grayvals[ind]
        w0 = w0 + hist[t]
        w1 = w1 - hist[t]
        std0 = np.std(grayvals[:ind])
        std1 = np.std(grayvals[ind:])
        std_total = w0 * std0 + w1 * std1
        std_inter.append((t, std_total))
    # print std_inter
    thresh = min(std_inter, key=lambda t: t[1])  # [thresold, std_inter]
    thresh = np.around(np.average([t for t, std in std_inter if std == thresh[1]]))
    return thresh


##### TRACKING RELATED FUNCTIONS #####

def alt_center(frame_gray, contours, cntInd):
    # Alternative center. Averages all the points inside the contour, including hollow spaces inside, instead of the exterior.
    mask = np.zeros(frame_gray.shape)
    cv2.drawContours(mask, contours, cntInd, 255, -1)  # Mask points inside contour
    mask = np.where(mask, frame_gray, 0)  # Get points inside contour
    indexes = np.array(np.nonzero(mask))
    center = np.around(np.average(indexes, axis=1)).astype(int)  # Average the points inside.
    return center[::-1]


def new_particle(label, Positions, countData):
    # New particle
    label += 1
    Positions[label] = [(None, None)] * (countData - 1)
    return label, Positions


def actualize_info(particle, Positions, center, activeParticles):
    # Adds the information of the nanomotor for that frame and draws it in the frame
    Positions[particle].append(center)
    activeParticles.append(particle)

    return Positions, activeParticles


def double_label_check(label, trackedParticle, Positions, center, activeParticles, countData):
    # Executes if two particles are recognized with the same label on the same frame
    cx, cy = center

    prevPos = [cent for cent in Positions[trackedParticle] if cent != (None, None)]
    if len(prevPos) > 2:
        # We have enough information to decide which is the correct one
        prevCenter = prevPos[-2]  # The second-to-last center appended to the list is one from the previous frame
        otherCenter = prevPos[-1]  # The last center appended to the list is from the same frame
        Dcurrent = np.linalg.norm((prevCenter[0] - cx, prevCenter[1] - cy))
        Dother = np.linalg.norm((prevCenter[0] - otherCenter[0], prevCenter[1] - otherCenter[1]))
        if Dcurrent < Dother:
            # Current is the real particle -> Delete the wrong one and give it a new label
            del Positions[trackedParticle][-1]
            label, Positions = new_particle(label, Positions, countData)
            Positions, activeParticles = actualize_info(label, Positions, otherCenter, activeParticles)

        else:
            # Current is another particle -> Therefore a new one
            label, Positions = new_particle(label, Positions, countData)
            trackedParticle = label

    else:
        # Not enough info to know which is the correct particle -> We give the current a new label
        label, Positions = new_particle(label, Positions, countData)
        trackedParticle = label

    return label, trackedParticle, Positions, activeParticles


##### DATA ANALYSIS FUNCTIONS #####

def mean_square(vector):
    # Input: vector with data
    # Output: mean square displacement given by MSD(p) = sum( (f(i+p)-f(i))**2)/total
    length = len(vector)
    intList = np.arange(1, length)  # intervals
    MSD = np.arange(1, length, dtype=float)  # To save the MSD values
    for interval in intList:
        intVector = [1] + [0] * (interval - 1) + [-1]  # Ex: for int = 3 you have [1,0,0,-1]
        # With "valid" you only take the overlapping points of the convolution
        convolution = np.convolve(vector, intVector, 'valid')
        MSDlist = convolution ** 2
        MSD[interval - 1] = np.average(MSDlist)
    return intList, MSD


def mean_square_displacement(xvector, yvector):
    # Input: vector with 2d positions in tuples
    # Output: mean square displacement given by MSD(p) = sum( (r(i+p)-r(i))**2)/total
    length = len(xvector)
    intList = np.arange(1, length)  # intervals
    MSD = np.arange(1, length, dtype=float)  # To save the MSD values
    for interval in intList:
        intVector = [1] + [0] * (interval - 1) + [-1]  # Ex: for int = 3 you have [1,0,0,-1]
        # With "valid" you only take the overlapping points of the convolution
        convolutionx = np.convolve(xvector, intVector, 'valid')
        convolutiony = np.convolve(yvector, intVector, 'valid')
        MSDlist = convolutionx ** 2 + convolutiony ** 2
        MSD[interval - 1] = np.average(MSDlist)
    return intList, MSD


def angular_auto_correlation(vector):
    l = len(vector)
    intList = np.arange(1, l)
    AAC = np.arange(1, l, dtype=float)
    AAC_dev = np.arange(1, l, dtype=float)
    for interval in intList:
        intVector = [1] + [0] * (interval - 1) + [-1]
        convolution = np.convolve(vector, intVector, 'valid')
        AACList = np.cos(np.deg2rad(convolution))
        AAC[interval - 1] = np.average(AACList)
        AAC_dev[interval - 1] = np.std(AACList)
    return intList, AAC, AAC_dev


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


# def parabola(x,a):
#     return a*x**2


def fit_parabola(xdata, ydata, guess):
    # "Guess" are intial guesses at the parameters. Can be "None"
    param, paramCOV = curve_fit(parabola, xdata, ydata, guess)
    return param, np.sqrt(np.diag(paramCOV))


def show_MSD(partPos, guess, showError):
    intervals, MSD = mean_square_displacement(partPos)  # Get MSD
    plt.plot(intervals, MSD, 'o')  # Plot MSD
    param, error = fit_parabola(intervals, MSD, guess)  # Fit parabola
    xfit = np.linspace(intervals[0], intervals[-1], 5 * len(intervals))
    yfit = parabola(xfit, param[0], param[1], param[2])
    name = 'y = %f.2 * x*x + %f.2 * x + %f.2' % (param[0], param[1], param[2])
    plt.plot(xfit, yfit, 'r', label=name)
    plt.legend()
    plt.show()

    if showError == True:
        print(param)
        print(error)


def save_data_raw(filename, Positions, Frames, unit_distance, unit_time):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for particle in Positions:
            X, Y = list(zip(*Positions[particle]))
            writer.writerow(['Particle', str(particle)])
            writer.writerow(['X (' + unit_distance + ')'] + list(X))
            writer.writerow(['Y (' + unit_distance + ')'] + list(Y))
            writer.writerow([unit_time] + Frames)
            writer.writerow('')


def Save_data(data, path):
    # Input: dictionaries with equal labels for particles
    with open(path + 'data.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Particle', 'Mean velocity modulus', 'Mean velocity vector'])
        for particle in data:
            info = data[particle]
            row = [particle, info[0], info[1]]
            writer.writerow(row)


def centered_difference(points, order):
    # Computes the first derivative of a set of points by applying the method of centered differences. Order can be 2, 4, 6 and 8
    Dij = Ddict[order]
    # [modified] error in py3(numpy1.12), change float to int
    # coef = Dij[order/2, :][::-1]
    # acc = (len(coef)-1)/2
    coef = Dij[order // 2, :][::-1]
    acc = (len(coef) - 1) // 2

    # Apply central difference
    deriv = np.convolve(points, coef, 'valid')
    deriv = np.hstack((acc * [0], deriv, acc * [0]))

    # Fill initial and final velocities
    deriv[0] = np.array([points[1] - points[0]])
    deriv[-1] = np.array([points[-1] - points[-2]])
    if order > 2:
        deriv[1] = np.array([points[2] - points[0]]) / 2.0
        deriv[-2] = np.array([points[-1] - points[-3]]) / 2.0
        if order > 4:
            coef4CD = Ddict[4][2, :]
            deriv[2] = np.dot(coef4CD, points[:5])
            deriv[-3] = np.dot(coef4CD, points[-5:])
            if order > 6:
                coef6CD = Ddict[6][3, :]
                deriv[3] = np.dot(coef6CD, points[:7])
                deriv[-4] = np.dot(coef6CD, points[-7:])

    return deriv


def get_startend_dict(Positions, infodict=False):
    # Creates Start/End dictionary from Positions information. If infodict = True, will assume input is "infoDict"
    if infodict:
        Positions = {p: Positions[p, info] for p, info in list(Positions.keys()) if info == 'position'}

    StartEnd = {}
    for particle in Positions:
        centers = Positions[particle]
        StartEnd[particle] = get_startend(centers)
    return StartEnd


def get_startend(data):
    isthere = [center != (None, None) for center in data]
    start = isthere.index(1)
    end = len(isthere) - 1 - isthere[::-1].index(1)
    return [start, end]


def interpolate(datapoints, startend, mode, angle=False):
    # Interpolates a list of datapoints. The missing ones are marked with None. Returns datapoints from "start" to "end"
    tt, data = list(zip(*[(tt, data) for tt, data in enumerate(datapoints) if data != None]))

    if angle == True:
        # Interpolation won't work unless we take care of 360->0 deg jumps. "datapoints" is assumed to be in degrees
        data = extend_angles(data)

    if len(tt) >= 4:
        finter = interp1d(tt, data, kind=mode)
        tt = list(range(len(datapoints)))[startend[0]:startend[1] + 1]
        return finter(tt)
    else:
        # Not enough points to interpolate: make constant interpolation
        datapoints = list(datapoints[startend[0]:startend[1] + 1])
        for t, d in enumerate(datapoints):
            if d != None:
                last = d
            if d == None:
                datapoints[t] = last
        return datapoints


def calculate_D(order):
    # Calculates the differentiation matrix D of arbitrary order n
    # For equidistant points x0, x1, ..., xn you can calculate the derivative of a set of points f(x) with the formula [f'(x0), f'(x1),...,f'(xn)] = 1/h * D * [f(x0), f(x1),...,f(xn)]

    # In our case h = 1 (one frame of difference)
    order = order + 1
    nodes = np.arange(1, order + 1)
    nodes_mask = np.ma.array(nodes, mask=False)

    # Baricentric weights
    weights = np.ndarray.astype(nodes, float)
    for i in range(order):
        nodes_mask.mask[i] = True
        weights[i] = 1.0 / np.prod(nodes[i] - nodes_mask)
        nodes_mask.mask[i] = False

    # Calculation of Dij
    D = np.empty((order, order), dtype=float)
    for i in range(order):
        for j in range(order):
            if i != j:
                D[i, j] = weights[j] / (weights[i] * (nodes[i] - nodes[j]))
            else:
                nodes_mask.mask[i] = True
                D[i, j] = np.sum(1.0 / (nodes[i] - nodes_mask)[np.invert(nodes_mask.mask)])
                nodes_mask.mask[i] = False

    return D


def predict_next(points):
    # Tries to predict the next position based on, at best, 4 previous points
    c_list = get_prev_consecutive_c(points, num=4)
    if len(c_list) > 1:
        X, Y = list(zip(*c_list))
        interX = interp1d(list(range(len(X))), X, kind='cubic')
        interY = interp1d(list(range(len(Y))), Y, kind='cubic')
        predicted_c = interX(len(points) + 1), interY(len(points) + 1)
        return [int(predicted_c[0]), int(predicted_c[1])]

    elif len(c_list) == 1:
        return c_list[0]

    else:
        print('predict_next error: no previous point to predict')


##### ANGLE-RELATED FUNCTIONS #####

def angle_difference(angle1, angle2, period=360):
    # Difference between angles (angle2 - angle1) taking into account periodicity in the 0-360 degree range
    angle2_extend = np.array([angle2, angle2 + period, angle2 + 2 * period])
    angle1 = angle1 + period
    diff_extended = angle2_extend - angle1
    diff_extended_abs = np.abs(diff_extended)
    diff = np.min(diff_extended_abs)
    diff_sign = np.sign(diff_extended[np.where(diff_extended_abs == diff)][0])
    return diff * diff_sign


def extend_angles(angles, period=360):
    if period == 360:
        return np.rad2deg(np.unwrap(np.deg2rad(angles)))
    elif period == 2 * np.pi:
        return np.unwrap(angles)
    #    #Make transitions between quadrants continous


#    #Example: [300, 350, 30, 80] -> [300, 350, 390, 430]
#    quadrant = 0
#    prevAng = angles[0]
#    angles_ext = np.array(angles)
#    for num,ang in enumerate(angles):
#        if num == 0:
#            continue
#
#        if abs(ang-prevAng) > 180:
#            #Angle in different quadrant
#            if ang < prevAng:
#                #One quadrant to the right [350, 40] -> [350, 400]
#                quadrant += 1
#            else:
#                #One quadrant to the left [20, 340] -> [20, -20]
#                quadrant -= 1
#        angles_ext[num] += (quadrant * period)
#
#        prevAng = ang
#    return list(angles_ext)

def ang_from_vec(vec, retInt=0):
    # Returns angle in degrees calculated from a vector
    ang = np.rad2deg(np.arctan2(vec[1], vec[0]))
    if retInt == 0:
        return np.around(ang % 360).astype(int)
    else:
        return np.around(ang % 360, decimals=retInt)


def vec_from_ang(ang, r=1):
    # Returns vector calculated from angle in radians
    return r * np.array([np.cos(ang), np.sin(ang)])


def rad2deg(rad):
    return np.rad2deg(rad)


def deg2rad(deg):
    return np.deg2rad(deg)


def rotate(point, angle):
    sin = np.sin(angle)
    cos = np.cos(angle)
    rotMatrix = np.array([[cos, -sin], [sin, cos]])
    point = np.array(point)
    return np.dot(rotMatrix, point)


def angmap180(ang):
    # Maps 0 to 360 angles to -180 to 180. Excludes -180. Example: [0,45,90,135,180,225,270,315,360] -> [0,-45,-135,180,135,90,45,0]
    if type(ang) == list or type(ang) == tuple:
        ang = np.array(ang)
    ang = ang - 180
    sign = sign2(ang)
    ang = sign * (180 - np.abs(ang))
    return ang


def sign2(x):
    # Sign function that returns 1 when x = 0
    sign = np.sign(x)
    if type(sign) == np.ndarray:
        sign[sign == 0] = 1
    elif sign == 0:
        sign = 1
    return sign


# Store D values
Ddict = {}
for order in range(2, 10, 2):
    Ddict[order] = calculate_D(order)


class DetectionFunctions(object):
    # Class that englobes all relevant functions for the detection class

    def __init__(self):
        pass

    #### IMAGE TREATMENT ####

    def smoothing(self, image):
        # Transform to grayscale
        if len(image.shape) == 3:
            # This way we can use the smoothing function with already gray images
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = image

        # Treat the image to reduce noise

        ###### CONTRAST ######
        if self.Vars['img_equalize_hist'] == True:
            frame_gray = cv2.equalizeHist(frame_gray)

        for j in range(self.Vars['img_clahe']):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            frame_gray = clahe.apply(frame_gray)

            ###### SMOOTHING ######
        if self.Vars['img_bilateral_d'] != 0:
            frame_gray = cv2.bilateralFilter(frame_gray, self.Vars['img_bilateral_d'], self.Vars['img_bilateral_color'],
                                             self.Vars['img_bilateral_space'])
        frame_gray = cv2.medianBlur(frame_gray, self.Vars['img_medianblur'])  # Good for eliminating white noise
        frame_gray = cv2.GaussianBlur(frame_gray, (self.Vars['img_gaussblur'],) * 2, 0)  # Good for general smoothing

        frame = frame_gray.copy()
        frame = gray2color(frame)

        return frame, frame_gray

    def morphology(self, image):
        # Applies different morphological operations to the image, which should be binary

        kernel = np.ones((self.Vars['img_close'],) * 2, np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Eliminate small errors
        kernel = np.ones((self.Vars['img_open'],) * 2, np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # Eliminate small errors

        return image

    #### GETTING PARTICLE INFORMATION ####

    def get_contour_information(self, M, extra=True, **kwargs):
        # Calculate centroid and other information of the particle (optional "extra" parameter triggers this calculation)
        # Returns the center
        area = M['m00']
        # Center from contours
        cx = int(round(M['m10'] / area))
        cy = int(round(M['m01'] / area))

        if 'ROIt' in kwargs:
            # If the "M" is computed for a ROI and not the full frame, the center is displaced
            cx += kwargs['ROIt'][0]
            cy += kwargs['ROIt'][1]

        center = np.array([cx, cy])

        if extra:
            for info in self.trackInfo[1:]:
                if info == 'angleJet':
                    _, ang = self.get_orientation(M)

                elif info == 'angleJanus':
                    ang = self.get_angle_janus(kwargs['frame_gray'], center)  # Analyze the image

                elif info == 'angleJetBubble':
                    [ang, points], (cx, cy) = self.get_angle_jet_bubble(center, kwargs[
                        'cnt'])  # "ang" includes the angle AND contour points
                    center = np.array([cx, cy])  # Center is modified
                    self.centerDict[((cx, cy), 'bubble_cnt')] = points

                elif info == 'angleZ':
                    ang = self.get_angleZ(kwargs['frame_gray'], kwargs['binary'], kwargs[
                        'cnt'])  # Get sum of grayscale values to approximate angle in Z direction

                self.centerDict[((cx, cy), info)] = ang

        if self.trackMode == 'bgextr' and self.bg_mode == 'Single Frame' and self.countData % self.Vars[
            'bg_actualize'] == 0:
            self.centerDict[((cx, cy), 'cnt')] = kwargs['cnt']

        return center

    def get_orientation(self, M):
        # Returns orientation in degrees based on the dict of moments "M"
        area = M['m00']

        mu20 = int(np.around(M['mu20'] / area))
        mu02 = int(np.around(M['mu02'] / area))
        mu11 = int(np.around(M['mu11'] / area))

        Mcov = np.array([[mu20, mu11], [mu11, mu02]])
        eigVal, eigVec = np.linalg.eig(Mcov)

        vec = eigVec[:, eigVal.argmax()]
        ang = ang_from_vec(vec)

        # # Faster but less exact
        # if mu20 != mu02:
        #     ang = 0.5 * np.arctan(2*mu11 / (mu20-mu02))
        #     ang = np.rad2deg(ang)
        #     vec = dfun.vec_from_ang(ang)

        # prevAngle,_ = get_prev(self.infoDict[particle,'angleJet'])
        # if prevAngle != None and 90 < abs(angle_difference(prevAngle, ang)) < 180:
        #     print prevAngle, ang
        #     ang = (ang-180) % 360 
        #     print ang

        return vec, ang

    def get_angle_janus(self, frame_gray, center):
        stdDict = {}  # {ang : standard deviation along the axis}
        shape = frame_gray.shape[::-1]
        for ang, vec in list(self.vecDict.items()):
            start = center - vec;
            end = center + vec
            start = bound(shape, start);
            end = bound(shape, end)
            linePoints = bresenham(start, end)  # Get points along the line
            values = frame_gray[linePoints]  # Get the grayscale values
            std = np.std(values)  # Standard deviation
            stdDict[ang] = std

        # PUT ALL ABOVE IN KEY?
        minAng = min(stdDict, key=stdDict.get)  # Angle with minimum standard deviation

        # Which of the two possible directions is correct? The brightest one!
        minAng += 90  # One of the sides is chosen arbitrarily
        vec = vec_from_ang((minAng) * (np.pi / 180), r=self.Vars['extra_partRadi'] * 0.5)
        testPoint1 = tuple(bound(shape, center + vec))[::-1]
        testPoint2 = tuple(bound(shape, center - vec))[::-1]
        if frame_gray[testPoint1] < frame_gray[testPoint2]:
            minAng += 180
        return minAng

    def get_angle_jet_bubble(self, center, cnt):
        r = 2 * self.Vars['extra_partRadi'];
        w = self.Vars['extra_partWidth']
        # Compare two lists of parallel points, and compute the distance between each of them. The point corresponding to the jet (rectangular) will have one of the most uniform distribution of distances
        cntLen = cnt.shape[0];
        indList = list(range(0, cntLen, w))
        indR_base = np.arange(r + w, r + w + r)  # From central point to the right
        indL_base = np.arange(r, 0, -1)  # From central point to the left
        std = [];
        pointsList = []
        for ind in indList:
            indR = indR_base + ind
            indL = indL_base + ind
            pointsR = np.take(cnt, indR, mode='wrap', axis=0)[:, 0, :]
            pointsL = np.take(cnt, indL, mode='wrap', axis=0)[:, 0, :]
            d = np.linalg.norm(pointsR - pointsL, axis=1)  # Distance between contour points
            d = d * (100 / max(d))  # Normalize
            std.append(np.std(d))  # Standard deviation
            pointsList.append([pointsR, pointsL])  # Contour points for linear regression

        # We use the points with least dispersion to do a linear regression
        tries = 2
        regInfo = np.zeros((tries, 3))  # Col1: index, Col2: slopes, Col3: funVal
        std = np.vstack((std, list(range(len(std))))).T  # Col1: std, Col2: stdIndex
        std = std[np.argsort(std[:, 0])][:tries, :]  # Col1: ordered std, Col2: stdIndex

        for ii in range(tries):
            tryInd = std[ii, 1].astype(int)

            if ii > 0 and std[ii, 0] - std[0, 0] > 2:
                regInfo = regInfo[:ii,
                          :]  # We ignore the points with standard deviations larger than the minimum std + 1
                break

            regInfo[ii, 0] = tryInd
            pointsRL = pointsList[tryInd]

            # Check convexity. Is the point in-between the two extremes inside the contour?
            p1 = pointsRL[0][-1, :];
            p2 = pointsRL[1][-1, :]
            pm = 0.5 * (p1 + p2)
            isConvex = (cv2.pointPolygonTest(cnt, tuple(pm), False) > -1)
            if isConvex == False:
                regInfo[
                    ii, 2] = -1  # If the pm is outside, the contour is concave and can't be the jet. A negative funVal will put this option to the back of the list
                continue

            funVal = np.array([[0, 0], [0, 0]]).astype(float)  # [[ slopeRight, RvalRight], [slopeLeft, RvalLeft]]
            for jj, points in enumerate(pointsRL):
                regSlope, regInter, regRval, refPval, regSTD = linregress(points)
                regRval = regRval ** 2  # Maybe change for standard deviation?
                funVal[jj] = [regSlope, regRval]

                # We calculate how good the regression is (average of Rval) and how similar the two sides are
            avRval = np.mean(funVal[:, 1])
            #            minInd = np.argmin(abs(funVal[:,0]))
            #            diffSlope = funVal[minInd,0]/funVal[1-minInd,0] #1-abs((maxSlope-minSlope)/maxSlope)
            fun = avRval  # + diffSlope
            # Store stuff in regression info matrix

            regInfo[ii, 1] = np.mean(funVal[:, 0])
            regInfo[ii, 2] = fun

            # The point with the highest linear correlation is selected
        headInd = indList[int(regInfo[np.argmax(regInfo[:, 2]), 0])]
        indexes = np.arange(headInd, headInd + r + w + r)
        points = np.take(cnt, indexes, mode='wrap', axis=0)[:, 0, :]
        center = np.mean(points, axis=0, dtype=int)

        # Ang taken from the obtained contour points of the jet head
        M = cv2.moments(points.reshape((points.shape[0], 1, 2)))
        _, ang = self.get_orientation(M)

        return [ang, points], center

    #    def get_angle_jet_bubble1(self,center,cnt):
    #        r = self.Vars['extra_partRadi']; w = self.Vars['extra_partWidth']/2
    #
    #        #1st step: get extreme points
    #        distances = np.linalg.norm(cnt - np.array(center), axis = 2)
    #        index1 = np.argmax(distances)
    #        extreme1 = cnt[index1,:,:][0] #Point farthest away from the center
    #
    #        distancesToExtreme = np.linalg.norm(cnt-extreme1, axis = 2)
    #        index2 = np.argmax(distancesToExtreme)
    #        extreme2 = cnt[index2,:,:][0] #Point farthest away from the first extreme
    #
    #        points = []
    #
    #        #2nd step: do a linear regression
    #        regInfo = np.array([[0,0],[0,0]]).astype(float) #[ [slope1, funVal1], [slope2, funVal2] ]
    #        print 'Start'
    #        for ii, (extr, ind) in enumerate([(extreme1, index1), (extreme2, index2)]):
    #            indRight = (np.arange(ind + w, ind + w + r*2)) #Points to the right of the extreme
    #            indLeft = (np.arange(ind - w - r*2, ind - w)) #Points to the left of the extreme
    #
    #            funVal = np.array([[0,0],[0,0]]).astype(float) #[[ slopeRight, RvalRight], [slopeLeft, RvalLeft]]
    #            for jj, indexes in enumerate([indRight, indLeft]):
    #                regPoints = np.take(cnt, indexes, mode = 'wrap', axis = 0)[:,0,:]
    #                points.append(regPoints)
    #                regSlope, regInter, regRval, refPval, regSTD = linregress(regPoints)
    #                regRval = regRval ** 2 #Maybe change for standard deviation?
    #                funVal[jj] = [regSlope, regRval]
    #
    #            #We calculate how good the regression is (average of Rval) and how similar the two sides are (difference of slopes)
    #            avRval = np.mean(funVal[:,1])
    #            minInd = np.argmin(abs(funVal[:,0]))
    #            diffSlope = funVal[minInd,0]/funVal[1-minInd,0] #1-abs((maxSlope-minSlope)/maxSlope)
    #            fun = avRval + diffSlope
    #
    #            regInfo[ii,0] = np.mean(funVal[:,0])
    #            regInfo[ii,1] = fun
    #            print 'slopes',[max(funVal[:,0]), min(funVal[:,0])], ' Rval', [funVal[0,1], funVal[1,1]], 'fun', [diffSlope, avRval, fun]
    #
    #               #### DELETE
    #            print regInfo
    #            f = self.frame.copy()
    #            pointsR, pointsL = pointsRL
    #            f[pointsR[:,1],pointsR[:,0]] = (0,255,0)
    #            f[pointsL[:,1],pointsL[:,0]] = (0,255,0)
    #            for ij in range(pointsR.shape[0]):
    #                cv2.circle(f, tuple(pointsR[ij,:]),1, (0,255,0),-1)
    #                cv2.circle(f, tuple(pointsL[ij,:]),1, (0,255,0),-1)
    #
    #
    #            cv2.imshow('ff',cv2.resize(f,None,fx = self.ff, fy = self.ff, interpolation = cv2.INTER_LINEAR))
    #            ch = cv2.waitKey() & 0xFF
    #            if ch == 27:
    #                sys.exit()
    #
    #        headInd = np.argmax(regInfo[:,1].ravel())
    #        #Get angle (from slope)
    #        slope = regInfo[headInd,0]
    #        ang = ang_from_vec([1,slope])
    #        #Get center (from contour)
    #        startInd = [index1,index2][headInd]
    #        indexes = np.arange(startInd - 2*r, startInd + 2*r)
    #        points = np.take(cnt, indexes, mode = 'wrap', axis = 0)[:,0,:]
    #        center = np.mean(points, axis = 0, dtype = int)
    #        startInd2 = [index2, index1][headInd]
    #        indexes2 = np.arange(startInd2 - 2*r, startInd2 + 2*r)
    #        points = np.vstack((points, np.take(cnt, indexes2, mode = 'wrap', axis = 0)[:,0,:]))
    #        return [ang,points], center

    #    def get_angleZ(gray, cnt):
    #        #Integrate the grayscale values inside the contour
    #        p_list = get_points_in_cnt(cnt)
    #        gray_val = gray[zip(*p_list)] #Grayscale values inside contour
    #        return np.sum(gray_val)

    def get_angleZ(self, gray, binary, cnt):
        x, y, w, h = cv2.boundingRect(cnt)  # For efficiency purposes we get the area around the contour
        binary = binary[y:y + h, x:x + w]

        #### METHOD 1 ####
        p_list = get_points_in_cnt(cnt)
        gray_val = gray[list(zip(*p_list))]  # Grayscale values inside contour
        std = np.std(gray_val)
        ratio2 = 0
        if std > 5:  # 12 if points in cnt do not include border, 15 otherwise
            t = otsu_thresh(gray_val)
            thresh_val = gray_val > t
            #            ratio2 = sum(thresh_val)/float(len(thresh_val))

            binary2 = np.zeros(binary.shape)
            p_list = np.array(p_list) - np.array([y, x])
            thresh_p = p_list[thresh_val]
            thresh_p = thresh_p
            binary2[list(zip(*thresh_p))] = 1
            kernel = np.ones((4, 4), np.uint8)
            binary2 = cv2.morphologyEx(binary2, cv2.MORPH_OPEN, kernel)  # Eliminate small errors
            ratio2 = np.sum(binary2 > 0) / float(len(thresh_val))

        return ratio2
        #### END METHOD 1 ####

    #
    #        ### REPRESENTATION ####
    #            plt.figure(3)
    #            plt.imshow(binary2)
    #
    ##        binary2 = np.zeros(binary.shape)
    ##        p_list = np.array(p_list) - np.array([y,x])
    ##        binary2[zip(*p_list)] = 1
    ##        plt.figure(3)
    ##        plt.imshow(binary2)
    #
    #        plt.figure(1)
    #        plt.hist(gray_val, bins = 15)
    #        plt.figure(2)
    #        plt.imshow(binary)
    #
    #        cv2.drawContours(gray, [cnt], -1, (255,0,0), 1)
    #        plt.figure(4)
    #        plt.imshow(gray[y:y+h, x:x+w])
    #        plt.show()
    #
    ##        print p_list, np.array(p_list), np.array(p_list).shape
    #        cv2.drawContours(gray, [cnt], -1, (255,0,0), 1)
    #        cv2.imshow('bin',gray[y:y+h, x:x+w])
    #        ch = cv2.waitKey() & 0xFF
    #        if ch == 27:
    #            sys.exit()
    #        cv2.destroyWindow('bin')
    ##        plt.close()
    #        ### REPRESENTATION ####
    #
    ##
    ##
    ##
    ##        #### METHOD 2 ####
    #        _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    #        hshape = hierarchy.shape
    ##        print hshape, binary.shape
    #        if hshape[1] == 1:
    #            return 0 #No interior contour
    #        elif hshape[1] == 2:
    #            if hierarchy[0,0,3] == -1: #Hierarchy is [Next, Previous, Child, Parent]. If Parent == -1, it means it is the outer contour.
    #                cnt_p = 0; cnt_c = 1
    #            else:
    #                cnt_p = 1; cnt_c = 0
    #
    #            M_p = cv2.moments(contours[cnt_p])
    #            M_c = cv2.moments(contours[cnt_c])
    #            if M_p['m00'] != 0:
    #                ratio = M_c['m00']/M_p['m00']
    #                return ratio
    #            else:
    #                return 0
    #
    #
    #        elif hshape[1] > 2:
    #            #print 'What' #More than two contours???
    #            return 0
    ##        #### END METHOD 2 ####

    #### HANDLE PARTICLE INFORMATION ####

    def new_particle(self):
        # New particle
        self.label += 1
        for info in self.trackInfo:
            if info in ['position', 'manual_angle']:
                self.infoDict[self.label, info] = [(None, None)] * (self.countData - 1)
            else:
                self.infoDict[self.label, info] = [None] * (self.countData - 1)

    def actualize_info(self, particle, data):
        # Adds the information of the nanomotor for that frame and draws it
        for num, info in enumerate(self.trackInfo):
            if info in ['angleJet', 'angleJetBubble']:
                data[num] = self.actualize_info_angleJet(particle, data[num], info)

            self.infoDict[particle, info].append(data[num])

        self.activeParticles.append(particle)

    def actualize_info_angleJet(self, particle, ang, info):
        # Angle is obtained with squared value: we force an arbitrary direction to avoid 180 deg jumps
        prevAngle, _ = get_prev(self.infoDict[particle, info])
        if prevAngle != None and 90 < abs(angle_difference(prevAngle, ang)):
            ang = (ang - 180) % 360
        return ang

    def draw_frame(self, frame, particle, data, thick=1, label=True, orientation=True, orientation_color=(255, 0, 255),
                   postp=False):
        for num, info in enumerate(self.trackInfo):

            if info == 'position':
                # Draws data in frame
                center = data[num]
                pt = (center[0] + 5, center[1] + 5)
                if label:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(particle), tuple(pt), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, center, thick, (0, 255, 0), -1)

            elif orientation and info == 'angleJet':
                ang = data[num]
                vec = vec_from_ang(ang * np.pi / 180)
                point = (np.array(center) + self.Vars['extra_partRadi'] * vec).astype(int)
                # draw_arrow(frame, center, tuple(point), np.pi/4, (255,0,255), thick)
                cv2.line(frame, center, tuple(point), orientation_color, thick)

            elif orientation and info == 'angleJanus':
                ang = data[num]
                shape = frame.shape[:2][::-1]

                # Half separator
                if ang in self.vecDict:
                    vec = self.vecDict[(ang + 90) % 180]
                else:
                    # For interpolated angles
                    vec = self.Vars['extra_partRadi'] * vec_from_ang((ang + 90) * np.pi / 180)
                point1 = center + vec;
                point2 = center - vec
                point1 = bound(shape, point1);
                point2 = bound(shape, point2)
                cv2.line(frame, tuple(point1), tuple(point2), (0, 255, 255), thick)

                # Orientation arrow
                point3 = center + vec_from_ang((ang) * (np.pi / 180), r=self.Vars['extra_partRadi'] * 1.5)
                point3 = bound(shape, point3)
                cv2.line(frame, tuple(center), tuple(point3), orientation_color, thick)

            elif orientation and info == 'angleJetBubble':
                ang = data[num]
                vec = vec_from_ang(ang * np.pi / 180)
                point = (np.array(center) + self.Vars['extra_partRadi'] * vec).astype(int)
                # draw_arrow(frame, center, tuple(point), np.pi/4, (255,0,255), thick)
                cv2.line(frame, center, tuple(point), orientation_color, thick)
                if postp == False:
                    points = self.centerDict[center, 'bubble_cnt']
                    frame[points[:, 1], points[:, 0]] = (0, 255, 0)


            elif orientation and info == 'angleVel':
                # Angle taken from velocity vector
                vec = vec_from_ang(data[num] * np.pi / 180)
                if 'angleJet' in self.trackInfo or 'angleJanus' in self.trackInfo:
                    r = self.Vars['extra_partRadi']
                else:
                    r = 20
                vec = r * vec
                point = (np.array(center) + vec).astype(int)
                color = np.array(orientation_color) + 125
                draw_arrow(frame, center, tuple(point), np.pi / 4, color_fix(color), thick)

            elif orientation and info == 'manual_angle':
                # Angle selected manually
                vec = vec_from_ang(data[num] * np.pi / 180)
                vec = self.Vars['manual_angle_r'] * vec
                point = (np.array(center) + vec).astype(int)
                cv2.line(frame, center, tuple(point), color_fix(orientation_color), thick)

            elif info == 'angleZ':
                ratio = np.around(data[num], decimals=3)
                pt = np.array(center) - [10, 10]
                cv2.putText(frame, str(ratio), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                # print particle, data[num]

    def double_label_check(self, trackedParticle, center):
        # Executes if two particles are recognized with the same label on the same frame
        cx, cy = center

        prevPos = [cent for cent in self.Positions[trackedParticle] if cent != (None, None)]
        if len(prevPos) > 2:
            # We have enough information to decide which is the correct one
            prevCenter = prevPos[-2]  # The second-to-last center appended to the list is one from the previous frame
            otherCenter = prevPos[-1]  # The last center appended to the list is from the same frame
            Dcurrent = np.linalg.norm((prevCenter[0] - cx, prevCenter[1] - cy))
            Dother = np.linalg.norm((prevCenter[0] - otherCenter[0], prevCenter[1] - otherCenter[1]))

            if Dcurrent < Dother:
                # Current is the real particle -> Delete the wrong one and give it a new label
                del self.Positions[trackedParticle][-1]
                self.new_particle()
                self.actualize_info(self.label, otherCenter)

            else:
                # Current is another particle -> Therefore a new one
                self.new_particle()
                self.trackedParticle = self.label

        else:
            # Not enough info to know which is the correct particle -> We give the current a new label
            self.new_particle()
            self.trackedParticle = self.label

    #### POST-PROCESSING OPTIONS ####

    def assign_colors(self, colorList):
        # Handles the assignement of colors to each particle. Needs global "Vars".
        data_color = {}  # { particle : color}
        if self.Vars['color_rand']:
            # Assign colors if the "randomized" option is selected
            colorIter = warped_iter(colorList)  # Warped Iterator of the colors
            for particle in self.Positions:
                color = colorIter.iter_next()
                color = color_fix(color)
                data_color[particle] = color

        else:
            color = colorList[0]
            color = color_fix(color)
            data_color = data_color.fromkeys(list(self.Positions.keys()), color)

        return data_color

    def get_velocity(self, order):
        # Calculates the velocities (vectorial and modulus) using a centered differentiation method of order "n", where n is the error of the method (2 goes as h**2, 4 goes as h**4, etc.)
        # Input must be dictionary {particle : [(x0,y0),(x1,y1),...,(xf,yf)]}
        VelInst = {}  # {particle : [v0, v1, ..., vf] } with v = vx**2 + vy**2
        # VelVec = {} # {particle : [(vx0,vy0),(vx1,vy1),...,(vxf,vyf)] }
        VelTotal = {}  # {particle : total distance / total time }

        conversion = (self.data_FPS / float(self.skipFrames))

        for particle in self.Positions:
            X, Y = list(zip(*self.Positions[particle]))
            if len(X) > 1:
                if len(X) > 2 * order:
                    vx = centered_difference(X, order)
                    vy = centered_difference(Y, order)

                    # self.VelVec[particle] = zip(vx,vy)
                    vel_mod = np.sqrt(vx ** 2 + vy ** 2)
                    VelInst[particle, 'position'] = vel_mod * conversion

                    for info in self.trackInfo[1:]:

                        if info == 'angleVel':
                            anglesVel = np.around(np.arctan2(vy, vx) * (180 / np.pi))  # Angle taken from velocity
                            self.infoDict[particle, info] = list(anglesVel)

                        if info in ['angleJet', 'angleJanus', 'angleJetBubble', 'manual_angle', 'angleVel']:
                            # Angular veloctiy
                            angles = self.infoDict[particle, info]
                            angles = list(extend_angles(angles))  # Continuous angles
                            w = centered_difference(angles, order)
                            VelInst[particle, info] = w

                        elif info == 'angleZ':
                            angles = self.infoDict[particle, info]
                            w = centered_difference(angles, order)
                            VelInst[particle, info] = w

                else:
                    # print 'Particle', particle, ': not enough points for computing instant velocity'
                    # VelVec[particle,'position'] = [(0,0)]
                    VelInst[particle, 'position'] = np.array([0])
                    for info in self.trackInfo[1:]:
                        VelInst[particle, info] = np.array([0])
                        if info == 'angleVel':
                            self.infoDict[particle, info] = [0] * len(X)

                # Calculate total velocity
                for info in self.trackInfo:
                    if info == 'position':
                        positions = np.array(self.Positions[particle]).transpose()
                        totDist = np.sqrt(np.sum(np.diff(positions) ** 2,
                                                 axis=0))  # Distances between every point sqrt( x**2 + y**2 )
                        totDist = np.sum(totDist)  # Total distance
                        VelTotal[particle, info] = (totDist / float(
                            len(X) - 1)) * conversion  # Total distance / Total time

                    elif info in ['angleJet', 'angleJanus', 'angleJetBubble', 'manual_angle', 'angleVel']:
                        angles = self.infoDict[particle, info]
                        diff = [angle_difference(angles[i], angles[i + 1]) for i in range(len(angles) - 1)]
                        totsum = sum(np.abs(diff))
                        VelTotal[particle, info] = (totsum / float(len(angles) - 1)) * conversion

                    elif info == 'angleZ':
                        angles = self.infoDict[particle, info]
                        totDist = np.sum(np.diff(angles))
                        VelTotal[particle, info] = (totDist / float(len(angles) - 1)) * conversion

            else:
                for info in self.trackInfo:
                    VelInst[particle, info] = np.array([0])
                    VelTotal[particle, info] = np.array([0])
                    if info == 'angleVel':
                        self.infoDict[particle, info] = [0]

        return VelInst, VelTotal

    def post_processing(self):
        # "self" must have the variables: "Vars", "infoDict", "countData", "trackInfo"
        self.saveInfoDict, self.saveCountData = self.infoDict, self.countData
        self.loopProcessing = True
        filecounterint = 0

        angleInfoList = ['angleJet', 'angleJanus', 'angleJetBubble', 'manual_angle']  # Possible angles from extra info
        angleInfoListExt = angleInfoList + ['angleVel']  # Previous list adding angle from velocity vector

        while (self.loopProcessing):

            self.infoDict, self.countData = copy.deepcopy(self.saveInfoDict), self.saveCountData

            totalParticles = len([key for key in list(self.infoDict.keys()) if 'position' in key])
            self.skipFrames = self.Vars['vid_skip']
            self.Frames = list(np.arange(1, self.countData + 1) * self.skipFrames + self.Vars['vid_start'])
            filecounter = '{:02}'.format(filecounterint)

            #### Choose particles to analyze ####

            root = tkinter.Tk()
            app2 = GUI_main.dataAnalysisGUI(root, Vars=self.Vars, infoDict=self.infoDict)

            root.mainloop()

            if app2.loopProcessing == False:
                # Go back to track menu
                break

            # Import variables
            for var in app2.variables:
                self.Vars[var] = app2.variables[var].get()

            print('Analyzing...')

            # Time units
            try:
                self.data_FPS = float(self.Vars['data_FPS'])
                self.unit_time = 'seconds'
                if self.data_FPS == 0.:
                    raise ValueError
            except ValueError:
                # Runs if Vars['data_FPS'] is not a number different than 0
                self.data_FPS = 1
                self.unit_time = 'frames'
            self.Frames = list(np.array(self.Frames) / self.data_FPS)

            # Distance units
            try:
                data_pix2mtr = float(self.Vars['data_pix2mtr'])
                self.unit_distance = 'microm'
                if data_pix2mtr == 0.:
                    raise ValueError
            except ValueError:
                data_pix2mtr = 1
                self.unit_distance = 'pixels'

            #### Purge the particles that have not been selected ####
            if not self.Vars['data_checkAll']:
                if len(self.Vars['data_ParticleList']) >= 1:

                    self.particleList = interpret_list(self.Vars['data_ParticleList'])

                    # Purge
                    for particle in range(1, totalParticles + 1):
                        if particle not in self.particleList:
                            for info in self.trackInfo:
                                del self.infoDict[particle, info]
            else:
                self.particleList = list(range(1, totalParticles + 1))

            ### Correct data ###
            # Eliminate entries (substitute as "None", which will be interpolated later)
            for p in app2.correct_deleteDict:
                for f in app2.correct_deleteDict[p]:
                    f = (f - self.Vars['vid_start']) / self.Vars['vid_skip']
                    for info in self.trackInfo:
                        if info == 'position':
                            self.infoDict[p, info][f] = (None, None)
                        elif info in angleInfoList + ['angleZ']:
                            self.infoDict[p, info][f] = None

            self.StartEnd = get_startend_dict(self.infoDict, infodict=True)
            # After eliminating possible erroneous entries, join selected particles
            joinDict = app2.correct_joinDict
            # Create chains
            starting_points = [p for p in list(joinDict.keys()) if p not in list(
                joinDict.values())]  # The start of a chain appears in keys but not in values
            chain_list = []
            for p in starting_points:
                chain = [p]
                while True:
                    p_next = joinDict[p]
                    chain.append(p_next)
                    if p_next not in list(joinDict.keys()):
                        break  # End of the chain, not in keys
                    else:
                        p = p_next

                chain_list.append(chain)

            for chain in chain_list:
                p1 = chain[0]
                for p2Ind in range(len(chain[1:])):
                    p2 = chain[p2Ind + 1]  # We make sure to go in order

                    p1start, p1end = self.StartEnd[p1]
                    p2start, p2end = self.StartEnd[p2]
                    sep = p2start - p1end - 1  # Number of frames separating both particles
                    for info in self.trackInfo:

                        if info == 'position':
                            # newdata = self.infoDict[p1,info][p1start:p1end+1] + [(None,None)]*sep + self.infoDict[p2,info][p2start:]
                            newdata = self.infoDict[p1, info][:p1end + 1] + [(None, None)] * sep + self.infoDict[
                                                                                                       p2, info][
                                                                                                   p2start:]

                        elif info in angleInfoList:

                            if info in ['angleJet', 'angleJetBubble']:
                                # newdatap1 = self.infoDict[p1,info][p1start:p1end+1]
                                newdatap1 = self.infoDict[p1, info][:p1end + 1]
                                newdatap2 = self.infoDict[p2, info][p2start:]
                                if abs(angle_difference(newdatap1[-1], newdatap2[
                                    0])) > 90:  # We assume the second particle had chosen a different sense for the jet orientation
                                    newdatap2swap = []
                                    for i in newdatap2:
                                        if i == None:
                                            newdatap2swap.append(i)
                                        else:
                                            newdatap2swap.append((i + 180) % 360)
                                    newdatap2 = newdatap2swap
                                newdata = newdatap1 + [None] * sep + newdatap2


                            else:
                                # newdata = self.infoDict[p1,info][p1start:p1end+1] + [None]*sep + self.infoDict[p2,info][p2start:]
                                newdata = self.infoDict[p1, info][:p1end + 1] + [None] * sep + self.infoDict[p2, info][
                                                                                               p2start:]

                        self.infoDict[p1, info] = newdata  # Substitute old data with extended one
                        del self.infoDict[p2, info]  # Delete joined particle

                    self.particleList.remove(p2)
                    self.StartEnd[p1] = [p1start, p2end]

            # We define self.Positions for comodity
            self.Positions = {part: self.infoDict[part, 'position'] for part in self.particleList}
            self.StartEnd = get_startend_dict(self.Positions)

            #### Saving data ####
            path_parts = self.Vars['filename'].split('/')
            path = '/'.join(path_parts[:-1]) + '/'
            path_parts[-1] = path_parts[-1][:-4]
            filename = path + path_parts[-1]
            filename_raw = filename + '_data_raw_' + filecounter + '.csv'

            self.save_data_raw(filename_raw)

            #### Interpolate missing points with cubic spline for automatic MSD ####
            interpMode = self.Vars['data_interp']
            self.fill_in(interpMode)

            if self.Vars['data_draw'] == True:
                drawPositions = self.Positions.copy()

            # Convert units
            for particle in self.particleList:
                pos = self.Positions[particle]
                self.Positions[particle] = list(zip(*list(zip(*(np.array(pos) / data_pix2mtr)))))

            if self.Vars['data_anglevel']:
                # Create space for new information: angle from vector velocity
                self.trackInfo.append('angleVel')

            #### Centered derivative to get velocities ####
            VelInst, VelTotal = self.get_velocity(self.Vars['data_order'])

            #### Saving data ####
            filename_inter = filename + '_data_inter_' + filecounter + '.csv'
            # [modified] error in py3
            # with open(filename_inter, 'wb') as csvfile:
            with open(filename_inter, 'w') as csvfile:
                writer = csv.writer(csvfile)

                # Parameter information
                sorted_keys = sorted(self.Vars.keys())
                sorted_values = [self.Vars[key] for key in sorted_keys]
                writer.writerow(['Parameter name:'] + sorted_keys)
                writer.writerow(['Parameter value:'] + sorted_values)
                writer.writerow('')

                # Error information
                writer.writerow(
                    ['Distance resolution:', str(np.around(1 / data_pix2mtr, decimals=3)) + ' ' + self.unit_distance])
                writer.writerow(
                    ['Time resolution:', str(np.around(1 / self.data_FPS, decimals=3)) + ' ' + self.unit_time])

                if 'angleJanus' in self.trackInfo:
                    ang_error = max(np.diff(np.sort(list(self.vecDict.keys()))))
                    writer.writerow(['Angle resolution:', str(ang_error) + ' degrees'])

                writer.writerow('')

                for particle in self.Positions:
                    # Title
                    writer.writerow(['Particle', str(particle)])

                    # Basic info
                    for info in self.trackInfo:
                        if info == 'position':
                            # Position
                            X, Y = list(zip(*self.Positions[particle]))
                            indexes = (self.StartEnd[particle][0], self.StartEnd[particle][1] + 1)
                            timeUnits = self.Frames[indexes[0]:indexes[1]]
                            writer.writerow(['Time (' + self.unit_time + ')'] + timeUnits)
                            writer.writerow(['X (' + self.unit_distance + ')'] + list(X))
                            writer.writerow(['Y (' + self.unit_distance + ')'] + list(Y))
                            # Velocity
                            inst = VelInst[particle, 'position']
                            writer.writerow(
                                ['Instantaneous velocity (' + self.unit_distance + '/' + self.unit_time + ')'] + list(
                                    inst))
                            writer.writerow(['Average vel modulus', str(np.average(inst))])
                            writer.writerow(
                                ['Total distance / Total time (' + self.unit_distance + '/' + self.unit_time + ')',
                                 str(VelTotal[particle, 'position'])])

                        elif info in angleInfoList + ['angleVel']:
                            extra_name = ''
                            if info == 'angleVel':
                                extra_name = 'from vel vector '
                            # Angle
                            angles = self.infoDict[particle, info]
                            writer.writerow(['Angle ' + extra_name + '(degrees)'] + [angmap180(ang % 360) for ang in
                                                                                     angles])  # Modulus
                            writer.writerow(
                                ['Angle ' + extra_name + '(degrees) (continuous)'] + [ang for ang in angles])
                            # Angular velocity
                            inst = VelInst[particle, info]
                            writer.writerow(
                                ['Inst angular velocity ' + extra_name + '(degrees/' + self.unit_time + ')'] + list(
                                    inst))
                            writer.writerow(
                                ['Average angular velocity ' + extra_name + '(degrees/' + self.unit_time + ')',
                                 str(np.average(np.abs(inst)))])
                            writer.writerow(
                                ['Total degrees ' + extra_name + '/ Total time (degrees/' + self.unit_time + ')',
                                 str(VelTotal[particle, info])])

                        #                        elif info == 'angleVel':
                        #                            #Angle
                        #                            anglesVel = self.infoDict[particle,info]
                        #                            writer.writerow(['Angle from vel vector (degrees)'] + [ang%360 - 180 for ang in anglesVel])
                        #                            writer.writerow(['Angle from vel vector (degrees) (continuous)'] + [ang-180 for ang in anglesVel])
                        #                            #Angular velocity
                        #                            inst = VelInst[particle,info]
                        #                            writer.writerow(['Inst angular velocity from vel vector (degrees/'+self.unit_time+')']+list(inst))
                        #                            writer.writerow(['Average angular velocity from vel vector (degrees/'+self.unit_time+')', str(np.average(np.abs(inst)))])
                        #                            writer.writerow(['Total degrees from vel vector/ Total time (degrees/'+self.unit_time+')', str(VelTotal[particle,info])])
                        #
                        elif info == 'angleZ':
                            # Angle
                            anglesZ = self.infoDict[particle, info]
                            writer.writerow(['Angle Z (degrees)'] + anglesZ)
                            # Angular velocity
                            inst = VelInst[particle, info]
                            writer.writerow(['Inst Z angle velocity (degrees/' + self.unit_time + ')'] + list(inst))
                            writer.writerow(['Average Z angle velocity (degrees/' + self.unit_time + ')',
                                             str(np.average(np.abs(inst)))])
                            writer.writerow(['Total angle Z degrees / Total time (degrees/' + self.unit_time + ')',
                                             str(VelTotal[particle, info])])

                            # Mean square displacements
                    incT, MSD = mean_square_displacement(X, Y)
                    incT = incT * (self.skipFrames / self.data_FPS)
                    writer.writerow(['Time displacement (' + self.unit_time + ')'] + list(incT))
                    writer.writerow(['MSD (' + self.unit_distance + '**2)'] + list(MSD))
                    for info in self.trackInfo[1:]:
                        if info in angleInfoList + ['angleVel']:
                            extra_name = ''
                            if info == 'angleVel':
                                extra_name = 'from vel vector'
                            angles = self.infoDict[particle, info]
                            # MSAD
                            _, MSAD = mean_square(angles)
                            writer.writerow(
                                ['Mean square angular displacement ' + extra_name + '(degrees**2)'] + list(MSAD))
                            # Angular auto correlation
                            if self.Vars['data_AAC'] == True:
                                _, AAC, AAC_dev = angular_auto_correlation(angles)
                                writer.writerow(['Angular auto-correlation ' + extra_name] + list(AAC))
                                writer.writerow(
                                    ['Angular auto-correlation ' + extra_name + 'standard deviation'] + list(AAC_dev))
                    writer.writerow('')

                writer.writerow('')

            print('Done')

            #### Draw trajectories ####
            if self.Vars['data_draw'] == True:

                # For choosing the video
                vid = cv2.VideoCapture(self.Vars['filename'])
                width, height = int(vid.get(3)), int(vid.get(4))

                # For saving the video
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Define codec
                filename_out = filename + '_nanoDetection_marker_' + filecounter + '.avi'
                out = cv2.VideoWriter(filename_out, fourcc, self.Vars['vid_FPS'],
                                      (width, height))  # Create VideoWriter object

                # Variables
                prev_count = self.countData  # Previous data count
                count = 0  # Counts frames
                self.countData = 0  # Counts analyzed frames (will vary from "count" when vid_skip > 1)
                recording = True
                frameMode = False
                imgTail = np.zeros((height, width, 3),
                                   dtype=np.uint8)  # Blank image used to save the trajectories before overlapping to the current frame
                imgTailAng = np.zeros((height, width, 3),
                                      dtype=np.uint8)  # Blank image used to save the angle vectors before overlapping to the current frame

                if np.any([info in self.trackInfo for info in angleInfoListExt]) == False:
                    # Orientation is selected even though no angle information is given: make it false
                    self.Vars['color_orientation'] = False

                    # Position iterators
                PosIter = {}
                for particle in self.Positions:
                    # The iterator will go through position indexes
                    PosIter[particle] = iter(list(range(len(drawPositions[particle]))))

                data_color = self.assign_colors(app2.colorList)
                # linepoints_set = set([]) # Set with the points that will be colored with imgTail #### Alt Overlap
                while (True):
                    count += 1  # Number of frames
                    ret, frame = vid.read()
                    if ret == False or self.countData == prev_count:
                        break
                    #                    frame = 255 * np.ones((height,width,3), dtype = np.uint8)

                    # Skip frames
                    if not (self.Vars['vid_start'] <= count <= self.Vars['vid_end']):
                        continue
                    if (count - self.Vars['vid_start']) % self.skipFrames == 0:
                        # Actualize the trajectory lines
                        # Save frame number
                        self.countData += 1
                        for particle in drawPositions:
                            start, end = self.StartEnd[particle]
                            color = tuple(data_color[particle])
                            # [modified]
                            # change <= end to <= end-1 to avoid index out of range
                            if start <= self.countData - 1 <= end - 1:
                                try:
                                    posIndex = next(PosIter[particle])
                                except StopIteration:
                                    print(drawPositions[particle])
                                    raise StopIteration

                                if start == self.countData - 1:
                                    point1 = point2 = drawPositions[particle][posIndex]
                                else:
                                    point1 = drawPositions[particle][posIndex - 1]
                                    point2 = drawPositions[particle][posIndex]
                                    if self.Vars['color_speed']:
                                        # [modified] line color changed with speed
                                        point3 = drawPositions[particle][posIndex + 1]
                                        dis1 = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
                                        dis2 = math.sqrt((point2[0] - point3[0]) ** 2 + (point2[1] - point3[1]) ** 2)
                                        speed1 = dis1 * self.data_FPS / float(self.Vars['vid_skip']) / float(
                                            self.Vars['data_pix2mtr'])
                                        speed2 = dis2 * self.data_FPS / float(self.Vars['vid_skip']) / float(
                                            self.Vars['data_pix2mtr'])
                                        point_inter = interp1d(np.linspace(0, 1, 2), [point1[1], point2[1]], "slinear")
                                        speed_inter = interp1d(np.linspace(0, 1, 2), [speed1, speed2], "slinear")
                                        xnew = np.linspace(point1[0], point2[0], int(dis1) + 2)
                                        ynew = point_inter(np.linspace(0, 1, int(dis1) + 2))
                                        speednew = speed_inter(np.linspace(0, 1, int(dis1) + 2))
                                        maxspeed = float(self.Vars['max_speed'])
                                        minspeed = float(self.Vars['min_speed'])
                                        print("speed: " + str(speed1))
                                        for i in range(len(xnew) - 1):
                                            speed = speednew[i]
                                            if speed > maxspeed:
                                                speed = maxspeed
                                            elif speed < minspeed:
                                                speed = minspeed
                                            color = hsv2rgb(*speed2hsv(speed, minspeed, maxspeed))
                                            cv2.line(imgTail, (int(xnew[i]), int(ynew[i])),
                                                     (int(xnew[i + 1]), int(ynew[i + 1])), color,
                                                     self.Vars['color_thick'])
                                    else:
                                        # [modified end]
                                        cv2.line(imgTail, point1, point2, color, self.Vars['color_thick'])
                                if self.Vars['color_orientation'] and self.Vars[
                                    'color_orientation_interval'] != 0 and self.countData % self.Vars[
                                    'color_orientation_interval'] == 1:
                                    # draw orientation every X frames
                                    data = [point2] + [self.infoDict[particle, info][posIndex] for info in
                                                       self.trackInfo[1:]]  # Particle info
                                    self.draw_frame(imgTailAng, particle, data, thick=self.Vars['color_thick'],
                                                    label=False,
                                                    orientation_color=color_invert(tuple(data_color[particle])),
                                                    postp=True)

                                # imgTail, points_set = draw_line(imgTail, point1, point2, color, self.Vars['color_thick']) #### Alt Overlap
                                # linepoints_set.update(points_set) #### Alt Overlap
                    # Overlap lines on frame
                    # frame = np.where(imgTail, imgTail, frame)
                    mask = imgTail > 0
                    frame[mask] = imgTail[mask]
                    if self.Vars['color_orientation']:
                        mask = imgTailAng > 0
                        frame[mask] = imgTailAng[mask]
                        # replace_points = zip(*linepoints_set) #### Alt Overlap
                    # frame[replace_points] = imgTail[replace_points] #### Alt Overlap

                    # We draw the center now so that it overlaps the lines
                    for particle in drawPositions:
                        start, end = self.StartEnd[particle]
                        if start <= self.countData - 1 < end:
                            posIndex = self.countData - start - 1
                            center = drawPositions[particle][posIndex]
                            data = [center] + [self.infoDict[particle, info][posIndex] for info in
                                               self.trackInfo[1:]]  # Particle info
                            self.draw_frame(frame, particle, data, thick=self.Vars['color_thick'],
                                            label=self.Vars['color_label'],
                                            orientation=self.Vars['color_orientation'],
                                            orientation_color=color_invert(tuple(data_color[particle])),
                                            postp=True)

                    # Show counter
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(count), (5, 15), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    # Fitting in monitor
                    if self.needsFitting == True:
                        frame_show = cv2.resize(frame, None, fx=self.ff, fy=self.ff, interpolation=cv2.INTER_LINEAR)
                    else:
                        frame_show = frame.copy()
                    cv2.imshow('frame', frame_show)

                    ch = cv2.waitKey(1) & 0xFF

                    if ch == 27:
                        # Exit
                        break
                    elif ch == ord('p') or frameMode == True:
                        # Pause
                        ch2 = cv2.waitKey() & 0xFF
                        if ch2 == 81 or ch2 == 83 or ch2 == 0:
                            # Next Frame
                            frameMode = True
                        else:
                            frameMode = False

                    if recording == True:
                        out.write(frame)

                out.release()
                vid.release()
                cv2.destroyAllWindows()

            # Repeat analysis?

            if self.Vars['data_anglevel']:
                del self.trackInfo[-1]

            root = tkinter.Tk()
            app3 = GUI_main.EndScreen(root)
            root.mainloop()

            self.loopProcessing = app3.loopProcessing
            self.loopTrack = app3.loopTrack
            filecounterint += 1

    def fill_in(self, mode):
        # Interpolates missing positions (if necessary)
        # Uses self.Positions, self.infoDict, self.particleList
        for particle in self.particleList:
            start, end = self.StartEnd[particle]
            X, Y = list(zip(*self.Positions[particle]))
            if None in X[start: end + 1]:
                # Only the particles with missing positions are interpolated
                for info in self.trackInfo:

                    if info == 'position':
                        interX, interY = interpolate(X, (start, end), mode), interpolate(Y, (start, end), mode)
                        if interX != 'wrong':
                            interX, interY = np.around(interX).astype(int), np.around(interY).astype(int)
                            newCenters = list(zip(interX, interY))
                        self.Positions[particle] = newCenters

                    elif info in ['angleJet', 'angleJanus', 'angleJetBubble', 'angleVel', 'manual_angle', 'angleZ']:
                        angle_wrap = True
                        if info == 'angleZ':
                            angle_wrap = False
                        interData = interpolate(self.infoDict[particle, info], (start, end), mode, angle=angle_wrap)
                        if info == 'angleZ':
                            interData = np.around(interData, decimals=6)
                        self.infoDict[particle, info] = list(interData)

            else:
                self.Positions[particle] = self.Positions[particle][start:end + 1]
                for info in self.trackInfo:
                    self.infoDict[particle, info] = list(extend_angles(
                        self.infoDict[particle, info][start:end + 1]))  # Extend the angles before post-processing

    def save_data_raw(self, filename):
        # Saves non-interpolated basic data.
        # Uses self.Positions, self.dictInfo, self.trackInfo, self.unit_distance, self.unit_time, self.Frames, self.particleList
        # [modified] error in py3
        # with open(filename, 'wb') as csvfile:
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for particle in self.particleList:
                X, Y = list(zip(*self.Positions[particle]))
                writer.writerow(['Particle', str(particle)])
                writer.writerow([self.unit_time] + self.Frames)
                writer.writerow(['X (' + self.unit_distance + ')'] + list(X))
                writer.writerow(['Y (' + self.unit_distance + ')'] + list(Y))
                for info in self.trackInfo[1:]:
                    if info in ['angleJet', 'angleJanus', 'angleJetBubble', 'manual_angle']:
                        writer.writerow([info + ' (degrees)'] + self.infoDict[particle, info])
                writer.writerow('')

        # #### Repeat analysis? ####
        # root = Tkinter.Tk()
        # GUI_main.exitScreen(root)
        # ret = root.mainloop()
        # import detection_main as det_main

        # if ret == 'main':
        #     det_main.start_program


# a = [1,2,3,4,5]


# seq = range(0,200,10)
# print seq, 100, get_closest(seq,100)

# warpedIter = warped_iter(a)

# print 'start'
# for i in range(10):
#     print warpedIter.iter_next()


# print warpedIter.next()
# print warpedIter.next()
# print warpedIter.next()
# print warpedIter.next()
# print warpedIter.next()

# def testfunction(t):
#     x = np.array(t)
#     #return 20*np.sin(x/10)
#     return x**2 + 0.3*np.exp(x/20) + 200*np.sin(x/10)

# def testfunctiond(t):
#     x = np.array(t)
#     #return 2*np.cos(x/10)
#     return 2*x + (0.3/20)*np.exp(x/50) + 20*np.cos(x/10)

# # # X = [1,2,3,5,7,9,12,15,18,20,22,24,25,26,27]
# a = 0; b = 50; N = 300; h = (b-a)/float(N)
# t = np.linspace(a,b,N)
# X = testfunction(t)
# Y = [1]*len(X)
# Positions = {1:zip(X,Y)}

# plt.plot(t, testfunctiond(t), label = 'real')
# for order in [4]:
#     velvec, velmod,_ = get_velocity2(Positions, order)
#     vx, vy = zip(*velvec[1])
#     vx, vy = np.array(vx)/h, np.array(vy)/h
#     plt.plot(t, vx, 'o', label = 'order ' + str(order))

# plt.legend(loc = 'best')
# plt.show()

if __name__ == '__main__':
    import detection_main

    detection_main.start_program()

NANOMOTOR TRACKING GUI

*Documentation for the nanomotor tracking GUI*

--------------------------- TRACKING METHODS -------------------------------

- AUTO tracking: all detected particles on-screen are automatically detected and analyzed. Only the ones below or above area limitations will be ignored.

- LIVE tracking: only the particles clicked by the user will be analyzed and tracked. However one MUST make sure that the particles can be detected a priori by using the preview feature. If no particle is detected in 10 frames the click will be ignored. If a previously detected particle is clicked, it will stop being followed.

- MANUAL tracking: no automatic detection is done. Position is specified by the user using the mouse. Angle can also be manually drawn.


--------------- GUI PARAMETERS (AUTO and LIVE tracking) ---------------------

####### Image treatment ########

Options for treating the image before starting nanomotor detection.

***Smoothing***
Good for eliminating image noise. If the values are too high it might blur the motors too much and make them harder to track.

-Median blur: substitutes a pixel value with the median of the surrounding pixels. Good for eliminating white noise. The number must be a positive integer, which determines the size of the matrix used for convolution. For example, a value of 3 will take the median of the surrounding nine values. A value of "1" will have no effect.

-Gaussian blur: substitutes a pixel value with the gaussian-weighted average of the surrounding pixels. Good for eliminating image defects in edges. The number must be a positive integer, which determines the size of the matrix used for convolution. For example, a value of 3 will take the gaussian average of the surrounding nine pixels. MUST BE AN ODD NUMBER. A value of "1" will have no effect.

-Bilateral filter: similar to gaussian blur, but maintains sharp borders un-blurred. It averages not only using spatial proximity, but also color proximity, so that object edges are respected.
	*Size: how much of an effect it will have in general (both in respecting edges and averaging over space)
	*Color: how much it will respect object edges (a larger value will blurr edges)
	*Space: how huge is the averaging area (a larger value will average over a lot of pixels)

***Histogram***
The histogram of an image is related to contrast, which is a way of measuring the range of colors of an image. Low contrast means all points have a colors in a small range, and the image will look very gray. Expanding it might make objects more easily distinguishable

-Histogram Equalizer: general equalizer, which expands the histogram to cover all gray-scale values. A gray image will be transformed to a fully black-and-white one. Will probably enhance noise, might cause saturation.

-Adaptive Histogram Equalizer: applies an histogram equalizer locally, taking into account the surrounding area. Avoids saturation and noise amplification.

***Morphological Operations***
Filters applied to the binary image.

-Opening: eliminates small white pixels, and clears connections between clusters of pixels. Complementary of closing.

-Closing: eliminates small holes (black pixels) inside white pixels. Closes gaps between clusters of pixels. Complementary of opening.

####### Detection and tracking options #######

Options related to nanomotor detection

-Search radius: the tracking is performed by searching in the area around a nanomotor for its previous position. The closest center from the previous frame that is inside this area is considered to be the nanomotor in its previous position. The search radius determines, in pixels, the radius of the circle in which the search is performed.

-(! IMPORTANT) Minimum area: when a shape is found in the image, the code must determine wether or not it corresponds to a nanomotor. "Minimum area" determines, in pixels, the minimum area of the detected shape needed to be considered a nanomotor. If the value is too low it might consider image noise as a nanomotor, if it's too big it might have difficulties in tracking small particles.

- Maximum area: same as before, but determines a maximum limit to shape area to be considered nanomotor.

- Tracking memory: to know wether a center from a previous time corresponds to a particle in the current frame both time and space closeness are checked. This value is equal to the amount of frames the tracking program will "remember" the position of a previous particle, meaning that after this time has passed it won't be tracked again. This can be useful to forget particles that went out of frame, or noise that dissapeared and crosses path with a real particle.


####### Video options #######

Options related to video viewing during the analysis

- Fit to screen: scales image to fit in a 1366x768 monitor, in case the size of the frame is bigger than that.

- Show binary image: shows a separate window with the binary image which results from taking the gradient and thresholding, after the optional closing and opening steps. The centers of the shapes in the binary images are assumed to be the nanomotors. Useful to check visually if the code is working correctly. For example: if the white blobs are too big in comparison to the real motors, maybe the thresholding parameter is set too low.

- Skip frame: only takes every nth frame to analyze. Example: if you put 5 it will take frame 1,5,10,15...

- Start/end frame: self-explanatory. Put 0 or -1 in "end" frame to take all of them.

- Video FPS: frames per second of the output "nanoDetection" videos. Can be a decimal number (eg: 23.4)

####### Extra information parameters #######

Parameters related to the extra information.

-Split touching particles: select for the program to attempt to differentiate two or more touching particles. Specialized in detecting circular objects.
	*Radi: expected radi of the individual particles. Must be accurate for the algorithm to work

-Particle radi: when angle information is requested, this indicates the radius of the particle to draw the vector. In the specific case of Janus particles this number must be accurate, since the orientation algorithm depends on it.
	*# of angles to test: when angle of Janus particles is requested, the algorithm tests different angles to find the correct one. The larger this number is, the more accurate the angle will be, but the code will run slower. Useless to go above 45, and if the radi is small (~10 pixels) there is no reason to make it larger than 15-20.


####### BUTTONS #######

- Set default: sets all parameters to their default values

- Set previous: sets all modified parameters to the values from the last sessions

- Go Back: go back to the previous menu

- Cancel: exit the program

- Start: begins analysis

####### PREVIEW #######

The preview feature let's you check the detection step before starting the tracking program, and actualizes every time a parameter is modified. Does NOT predict the particle tracking step, only the particle detection step. The number labels are assigned arbitrarily.

- Functionality: if opened in a command prompt, every time this windows is opened the following will appear in the output:
	*Time to preview: time in seconds that it took to open the preview
	*Counted Frames: number of frames that the video has, as counted by the tracking software (which is the correct value)
	*Counted Frames by OpenCV: the library OpenCV allows to check the number of frames directly. Sometimes it's incorrect, and there will be discrepancies with the real number of frames. If this number is the same as the previous "Counted Frames" value, the preview feature will actualize very fast. Otherwise the algorithm must iterate over the entire video every time a different frame is selected.

- Hotkeys: use the arrow keys to navigate over the frames. Press enter to manually actualize the image.

- Actualize button: when the frame number is changed using the scrollbar the window must be updated manually by pressing this button.


 --------------------- METHOD-SPECIFIC PARAMETERS ---------------------

####### Gradient options #######

Since the program must track both black and white particles, image gradients are used for detection. This section specifies parameters related to the way the gradient is calculated.

- (! IMPORTANT) Threshold: threshold used on the gradient image. Intuitively, it determines how different a nanomotor must be from its surrounding pixels to be detected by the program. Too low of a value will make the tracking very susceptible to noise, too high of a value will make it hard to detect faint nanomotors. Can be any positive number, and can go as high as 4000.

- Derivation order: order of the derivative calculated for the gradient.

- Kernel size: size of the matrix used in the convolution to get the gradient. Can only be -1, 1, 3, 5 or 7. "-1" is a special kernel similar to "3" but more resistant to noise.


####### Absolute thresholding options ########

Every pixel in the image with a grayscale value below the threshold is assigned a 0 and every pixel above is assigned a 1 (or viceversa). The nanomotor shapes are taken then from the binary image.

- Black nanomotors: if the particles that must be tracked are black on white background check this value. The program will do then inverse thresholding: only the pixels BELOW the threshold will be assigned a 1 and analyzed as posible particles.

- (! IMPORTANT) Threshold: global threshold used to binarize the image

####### Thresholding options ########

Similar to absolute thresholding, but the threshold value is assigned locally to every pixel using an average of its surroundings.

- (! IMPORTANT) Block size: size of the neighborhood used to compute the local threshold value

- (! IMPORTANT) Subtracting constant: the algorithm considers only the pixels with value "p" such that: "p - C > threshold", where "C" is the subtracting constant. Small noise might make background points trigger the algorithm (for example, in a neighborhood with mostly grayscale values of 100, a pixel with 101 could be above the average without being too different). The subtracting constant makes sure that only points that are very different from its surroundings get selected. Usually a constant around 5 is enough.

- Type of average: how the threshold value is calculated. Can be normal mean (no weights) or gaussian (using weights from a normal distribution, pixels contribute less as they are farther away)

- Black nanomotors: wether or not the particles to be detected are darker than the background.+

####### Background subtraction options ########

In bg extr mode every frame is compared with a reference image, and only the pixels that are different enough when compared with this reference image are selected, based on the "threshold" parameter. The reference image is an approximation of the background of the video.

- (! IMPORTANT) Threshold: minimum difference in gray-scale pixel value between the frame and the reference image for a pixel to be considered part of the foreground, and be detected. Usually around 30 is fine.

There are three ways of calculating the reference image:

- Single Frame: the reference image is a single frame taken from the video. Since it might contain particles in it, using it as reference might create "ghost" detections. The positions of these "ghost" particles are patched out dynamically with the video, thanks to the fact that they stay completely still in comparison the real particles.
	*Reference frame: number of the frame used as reference
	*Reference update: indicates how often a check on the reference image is done to eliminate "ghost" particles. Must be enough to let real particles move away from their initial position (in case the reference frame is the first one)
	*Background color: the "ghost" particles are differentiated using two characteristics: their null movement, and their color. The parameter "background color" indicates the grayscale pixel value above which background pixels usually are. Detected particles that don't move and have an average pixel value above this parameter are considered "ghosts" and are patched out.

- Total Average: the reference image is built by averaging the entire video. Can't work with live-feed cameras. Since preparing this average takes some time, the last instance in which it was calculated is saved, so that if the video is not changed it can be loaded again almost instantaneously.

- Dynamic Average: works best for live-feed cameras. The reference image is built on-the-run by averaging the previous frames using the following formula: ref = ref * alpha + (1-alpha) * frame. "ref" is the reference image, "frame" is the current frame and "alpha" is a number between 0 and 1.
	*Alpha: number that dictates how relevant previous frames are. The closer to 1 it is, the less important newer frames are. Usually a value above 0.9 is recommended.


--------------------- POST-PROCESSING OPTIONS ---------------------

Here you can select the particles you want analyzed for the excel files, and optionally draw their trajectories.

- Analyze entry: put the label of the particles you want to analyze as a comma-separated list. For example: "1, 2, 10, 43". Or check "all" to select all of them. Using a double dot selects all particles in-between the two numbers. For example: "2:5" is equivalent to "2,3,4,5". Can be used together with commas.

####### Drawing options #######

- Draw trajectory: runs the original video again but with the trajectory of the selected particles drawn in video time.

- Thickness: thickness, in pixels, of the drawn trajectories and angle vectors (if specified

- Color: color of the trajectory, can be specified with its RGB values. The selected color can be seen on the right.

- Randomize: choose a random set of colors in a random order. It's possible to specify an amount. By clicking on the color image a new set is created.

- Show particle label: if selected, the particle label is shown at the side

- Show orientation: if selected, orientation vectors are drawn on the image
	* Every "X" frames: if "0", only the current orientation is drawn. If larger than 0, this parameters specifies the interval at which the orientation vector is drawn on the image.

-Change color with speed: 选中后，运动轨迹颜色随速度变化，速度从minimum speed 变到 maximum speed，颜色从HSV空间(240, 1, 1)(蓝色)渐变为(0, 1, 1)(红色)（1.6.1.4版本加入）

####### Data processing options #######

Options related to interpolation from data

- Unit conversion: You can optionally give the FPS and the pixels per micrometer ratio to convert the data units. If you don't have them, leaving the entry blank or with a '0' will give the data in pixels and frames. Accepts decimal numbers in the form 1.2345 (NOT 1,234)

- FPS BUTTON: tool for automatically calculating the FPS. The TOTAL seconds of the video and the TOTAL frames of the video must be given. The number of frames can be calculated automatically by clicking the button next to the entry.

- Interpolation method for missing points: when a particle is missing for some frames the program tries to estimate the positions based on its previous and future locations. There are different methods available:

	*Linear: simply calculates a line between the two points that must be interpolated. Very fast to calculate, good for slow particles.
	*Cubic: cubic spline interpolation. Tries to approximate cubic polynomials with continous functions piece-wise. Very slow to compute, preferable for fast particles (estimated positions are more smooth and natural)
	*Quadratic: quadratic spline interpolation. Same as before but with quadratic polynomials, not as accurate as cubic.
	*Slinear: linear spline interpolation. Same as before but with linear polynomials only, not very accurate.
	*Nearest: for every missing point simply takes the data point that's closest in time. Very fast to calculate but it's not a good estimate.
	*Zero: no estimation

- Velocity precision: the velocity at every point is calculated by taking the derivative of the positions, which is approximated from the  datapoints using centered difference methods. These methods always have a certain error margin with respect to the real values, represented by O(h^n), where "h" is the time difference between the datapoints and "n" is the order of error. In the case of videos where "h" represents seconds between each frames, if n = 2 then increasing the number of frames tenfold will increase the accuracy of the velocity calculation by 100. Available orders are O(h²), O(h⁴), O(h⁶) and O(h⁸).

- Compute angle from velocity: if selected, additional angle information will be computed from the approximated instantaneous velocity. If the particles are slow, use the "skip parameters" option so that this value is more accurate. CAN BE USED TOGETHER WITH ANY OTHER TYPE OF ANGLE INFORMATION.

--------------------- MANUAL TRACKING ---------------------

Using this tracking method no detection is performed, the user draws the trajectory himself.

- Start/End frame: self-explanatory

- (! IMPORTANT)	Skip frames: imilar to the option in the auto/live tracking menus. Skips a certain amount of frames, and user input is only asked in non-skipped ones. Very important to select an appropriate value here. If it's too low the user might need to manually select redundant positions, if too high the trajectory might not be accurate.

- (! IMPORTANT) Hotkeys: position is selected by pushing left mouse button. Angle (if requested) is drawn by dragging the mouse, without releasing the left button, after selecting a position. "Backspace" eliminates the last input and goes back to a previous position. "l" toggles drawing the previous trajectory lines.

- Show lines: if selected, previously selected lines are drawn. Can be toggled by clicking "l" during the tracking, in case it obstructs the view

- Fit to Screen: self-explanatory

- Indicate Orientation: if selected, the user must draw the orientation along with the position. This is done in the following way: when the left mouse button is pushed, the position is saved. Then, without releasing, the user must move the mouse to draw an appropriate arrow that represents the angle of the nanomotor. When the mouse button is released the program saves both the position and the angle, and goes to the next frame.

- Number of stored frames: indicates how many previous positions are stored. This allows to go back to a previous state by pressing "backspace", and correcting mistakes.

####### MANUAL TRACKING HOTKEYS #######

- ESC: exits the video (all the information)

- BACKSPACE: deletes last selected point. Can go back as far as "stored frames" allows. If a particle is deleted with correction, the program will automatically go to the last position of the previous one.

- "n": starts tracking a new particle.

- "q": only AFTER clicking "n", move backwards one step without correcting.

- "w": only AFTER clicking "n", move forwards one step.

- "l": ON/OFF for drawing tracking lines.

- "t": increases line thickness.

- "y": decreases line thickness.

- "c": randomizes colors.


--------------- VIDEO CONTROLS ---------------------

-Press "ESC" to exit the video (all the information up to the current frame will be saved)
-Press "p" to pause the video
-While the video is paused, press the right or left arrow to advance one frame
-Press "r" to toggle resize (in case it's activated)
-Press "s" to save the current frame without labels






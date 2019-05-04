import sys  # For sys.exit and others
import os  # For directory finding
import configparser  # For writing and reading configuration files
import time  # Count time, delay
import copy
import random #Create random numbers, lists, etc.
import numpy as np  # Arrays
from PIL import Image, ImageTk
import cv2  # OpenCV
import tkinter
import tkinter.filedialog  # GUI ask for filename
import tkinter.ttk  # GUI interface
import detection_functions as dfun


class mainGUI(tkinter.Frame):
    
    trackMethodTitle = {'auto': 'Automatic ', 'live': 'Live ', 'manual': 'Manual '}

    def __init__(self, root, *args):

        # Variable declaration
        self.img_medianblur = tkinter.IntVar()  # Kernel size of median blur method
        self.img_gaussblur = tkinter.IntVar()  # Kernel size of gaussian blur method
        self.img_bilateral_d = tkinter.IntVar()  # Kernel size of bilateral blur method
        self.img_bilateral_color = tkinter.IntVar()  # Relative value of color proximity in bilateral method
        self.img_bilateral_space = tkinter.IntVar()  # Relative value of space proximity in bilateral method
        self.img_equalize_hist = tkinter.BooleanVar()  # Equalize histogram
        self.img_clahe = tkinter.IntVar()  # Number of times the adaptive histogram equalizer (clahe) filter is applied
        self.img_open = tkinter.IntVar()  # Kernel size of "opening" morphological operation
        self.img_close = tkinter.IntVar()  # Kernel size of "closing" morphologcal operation
        self.trk_radi = tkinter.IntVar()  # Tracking radius (in pixels)
        self.trk_minArea = tkinter.IntVar()  # Minimum area of nanomotors (in pixels)
        self.trk_maxArea = tkinter.IntVar()  # Maximum area of nanomotors (in pixels)
        self.trk_memory = tkinter.IntVar()  # Number of frames for a gone particle to be forgotten
        self.vid_contour = tkinter.BooleanVar()  # True if an additional video with contours is shown
        self.extra_partRadi = tkinter.IntVar()  # Approximate length in pixels of the motors
        self.extra_partWidth = tkinter.IntVar()  # Approximate width in pixels of a jet (only used in jet+bubbles)
        self.extra_ang_ticks = tkinter.StringVar()  # Number of tested angles for "angleJanus"
        self.extra_split = tkinter.BooleanVar()  # If true the code will attempt to differentiate touching particles
        self.extra_split_r = tkinter.IntVar()  # Approximate radius of the particles, used in splitting
        self.__init_basic__()

        self.variables = self.create_tkinter_dict()

        self.root = root
        self.isPreviewOn = False
        self.trackInfo = args[0]
        self.trackMethod = args[1]
        self.isCv2Accurate = False

        self.default_values.update({
            'img_medianblur': 1,
            'img_gaussblur': 3,
            'img_open': 1,
            'img_close': 1,
            'img_bilateral_d': 6,
            'img_bilateral_color': 75,
            'img_bilateral_space': 75,
            'img_equalize_hist': False,
            'img_clahe': 0,
            'trk_radi': 40,
            'trk_minArea': 30,
            'trk_maxArea':3000,
            'trk_memory':40,
            'vid_contour':True,
            'extra_partRadi': 10,
            'extra_partWidth': 5,
            'extra_ang_ticks': 'MAX',
            'extra_split': False,
            'extra_split_r': 10})

        self.N, self.S, self.E, self.W = tkinter.N, tkinter.S, tkinter.E, tkinter.W
        
        # Bind keys
        root.bind('<Return>', self.exit_task)  # Binds enter to the start button
        # root.bind('<s>', self.exit_task) # Binds enter to the start button

        root.bind('<Escape>', lambda *args:sys.exit()) # Binds enter to the exit button
        # root.bind('<c>', lambda *args:sys.exit()) #Binds enter to the exit button

        # root.bind('<BackSpace>', self.go_back) #Binds enter to the go back button
        # root.bind('<g>', self.go_back) #Binds enter to the go back button

        # Main frame #
        self.mainframe = tkinter.ttk.Frame(root, padding=(10, 10, 10, 5))
        self.videolbl = tkinter.ttk.Label(self.mainframe, text='Choose video:')
        self.entryVideo = tkinter.ttk.Entry(self.mainframe, textvariable=self.filename, width=40)
        self.videoButton = tkinter.ttk.Button(self.mainframe, text='Search...', command=self.get_filename)
        self.acceptButton = tkinter.ttk.Button(self.mainframe, text='Start', command=self.exit_task)
        self.acceptButton.state(['active'])
        self.cancelButton = tkinter.ttk.Button(self.mainframe, text='Cancel', command=lambda: sys.exit())
        self.previewButton = tkinter.ttk.Button(self.mainframe, text='Preview', command=self.show_preview)
        self.goBackButton = tkinter.ttk.Button(self.mainframe, text='Go back', width=10, command=self.go_back)

        self.valuebuttonframe = tkinter.ttk.Frame(self.mainframe, padding=(1, 1, 1, 1), relief='sunken')
        self.defaultButton = tkinter.ttk.Button(self.valuebuttonframe, text='Set default', command=self.set_default)
        self.previousButton = tkinter.ttk.Button(self.valuebuttonframe, text='Set previous', command=self.read_previous_config)
        
        # Scrollable entry
        self.entryVideo.focus()
        self.entryVideoScroll = tkinter.ttk.Scrollbar(self.mainframe, orient=tkinter.HORIZONTAL, command=self.scrollHandler)
        self.entryVideo['xscrollcommand'] = self.entryVideoScroll.set
        self.entryVideo.xview_moveto(1)

        # Image treatment options #
        self.imgframe = tkinter.ttk.Labelframe(self.mainframe, text=' Image treatment options: ', padding=10, relief='sunken')
        self.entryMedian = tkinter.Spinbox(self.imgframe, from_=1, to=100, increment_=2, textvariable=self.img_medianblur, width=2)
        self.entryGaussian = tkinter.Spinbox(self.imgframe, from_=1, to=100, increment_=2, textvariable=self.img_gaussblur, width=2)
        self.labelMedian = tkinter.ttk.Label(self.imgframe, text='Median blur')
        self.labelGaussian = tkinter.ttk.Label(self.imgframe, text='Gaussian blur')
        
        self.checkEqualizeHist = tkinter.ttk.Checkbutton(self.imgframe, text='Apply Histogram Equalizer', variable=self.img_equalize_hist, onvalue=True, offvalue=False)
        self.entryClahe = tkinter.Spinbox(self.imgframe, from_=0, to=100, increment_=1, textvariable=self.img_clahe, width=2)
        self.labelClahe = tkinter.ttk.Label(self.imgframe, text='Apply adaptive Hist. Eq.')
        
        self.entryOpen = tkinter.Spinbox(self.imgframe, from_=1, to=10000, textvariable=self.img_open, width=2)
        self.entryClose = tkinter.Spinbox(self.imgframe, from_=1, to=10000, textvariable=self.img_close, width=2)
        self.labelOpen = tkinter.ttk.Label(self.imgframe, text='Opening size')
        self.labelClose = tkinter.ttk.Label(self.imgframe, text='Closing size')
        self.bilateralframe = tkinter.ttk.Label(self.imgframe, padding=10)
        self.labelBilateral = tkinter.ttk.Label(self.bilateralframe, text='Bilateral filter (size / color / space)')
        self.entryBilateralD = tkinter.Spinbox(self.bilateralframe, from_=0, to=10000, textvariable=self.img_bilateral_d, width=2)
        self.entryBilateralColor = tkinter.ttk.Entry(self.bilateralframe, textvariable=self.img_bilateral_color, width=3)
        self.entryBilateralSpace = tkinter.ttk.Entry(self.bilateralframe, textvariable=self.img_bilateral_space, width=3)

        ##### Detection and tracking options #####
        self.trkframe = tkinter.ttk.Labelframe(self.mainframe, text=' Detection and tracking options: ', padding=10, relief='sunken')
        self.entryRadi = tkinter.ttk.Entry(self.trkframe, textvariable=self.trk_radi, width=4)
        self.entryMinArea = tkinter.ttk.Entry(self.trkframe, textvariable=self.trk_minArea, width=4)
        self.entryMaxArea = tkinter.ttk.Entry(self.trkframe, textvariable=self.trk_maxArea, width=4)
        self.labelRadi = tkinter.ttk.Label(self.trkframe, text='Search radius')
        self.labelMinArea = tkinter.ttk.Label(self.trkframe, text='Minimum area for detection')
        self.labelMaxArea = tkinter.ttk.Label(self.trkframe, text='Maximum area for detection')
        self.labelMemory = tkinter.ttk.Label(self.trkframe, text='Memory frames')
        self.entryMemory = tkinter.ttk.Entry(self.trkframe, textvariable=self.trk_memory, width=4)

        ##### Video options #####
        self.videoframe = tkinter.ttk.Labelframe(self.mainframe, text= ' Video options: ', padding = 10, relief = 'sunken')
        self.checkFit = tkinter.ttk.Checkbutton(self.videoframe, text = 'Fit to screen', variable = self.vid_fit, onvalue = True, offvalue = False)
        self.checkContour = tkinter.ttk.Checkbutton(self.videoframe, text = 'Show binary image', variable = self.vid_contour, onvalue = True, offvalue = False)
        self.labelSkip = tkinter.ttk.Label(self.videoframe, text = 'Skip frames ')
        self.entrySkip = tkinter.ttk.Entry(self.videoframe, textvariable = self.vid_skip, width = 4)
        self.frameVidStart = tkinter.ttk.Frame(self.videoframe, padding = (0,0,0,0))
        self.labelVidStart = tkinter.ttk.Label(self.frameVidStart, text = 'From frame ')
        self.entryVidStart = tkinter.ttk.Entry(self.frameVidStart, textvariable = self.vid_start, width = 5)
        self.frameVidEnd = tkinter.ttk.Frame(self.videoframe, padding = (0,0,0,0))
        self.labelVidEnd = tkinter.ttk.Label(self.frameVidEnd, text = ' to frame ')
        self.entryVidEnd = tkinter.ttk.Entry(self.frameVidEnd, textvariable = self.vid_end, width = 5)
        self.labelFPS = tkinter.ttk.Button(self.videoframe, text = 'Video FPS', command = lambda :self.calculate_FPS_window('vid'))
        self.entryFPS = tkinter.ttk.Entry(self.videoframe, textvariable = self.vid_FPS, width = 4)


        ##### Extra information parameters #####
#        if len(self.trackInfo) > 1:
        self.extraframe = tkinter.ttk.Labelframe(self.mainframe, text = ' Extra information parameters: ', padding = 10, relief = 'sunken')
        self.extraframe.grid(column = 0, row = 7, rowspan = 2, columnspan = 3, sticky = (self.N,self.S,self.E,self.W), pady = (10,0))

        self.checkSplit = tkinter.ttk.Checkbutton(self.extraframe, text = 'Split circular touching particles', variable = self.extra_split, onvalue = True, offvalue = False)
        self.labelSplitR = tkinter.ttk.Label(self.extraframe, text = 'Particle radi for splitting')
        self.entrySplitR = tkinter.Spinbox(self.extraframe, from_=1, to=10000, textvariable = self.extra_split_r, width = 3)
        self.checkSplit.grid(column = 0, row = 0, columnspan = 2, pady = 3, padx = (10,0), sticky = self.W)
        self.labelSplitR.grid(column = 2, row = 0, pady = 3, padx = (10,0), sticky = self.W)
        self.entrySplitR.grid(column = 3, row = 0, pady = 3, padx = (2,5), sticky = self.E)

        if 'angleJet' in self.trackInfo or 'angleJanus' in self.trackInfo or 'angleJetBubble' in self.trackInfo:
            self.labelExtraRadi = tkinter.ttk.Label(self.extraframe, text = 'Particle radi')
            self.entryExtraRadi = tkinter.Spinbox(self.extraframe, from_=1, to=10000, textvariable = self.extra_partRadi, width = 3)
            self.labelExtraRadi.grid(column = 0, row = 1, pady = 3, padx = (10,0), sticky = self.W)
            self.entryExtraRadi.grid(column = 1, row = 1, pady = 3, padx = (3,5), sticky = self.W)

            if 'angleJanus' in self.trackInfo:
                self.labelExtraTicks = tkinter.ttk.Label(self.extraframe, text = '# of angles to test')
                self.entryExtraTicks = tkinter.ttk.Combobox(self.extraframe, textvariable = self.extra_ang_ticks, width = 4)
                self.extra_partRadi.trace('w', self.set_extra_maxticks)
                self.labelExtraTicks.grid(column = 2, row = 1, pady = 3, padx = (10,0), sticky = self.W)
                self.entryExtraTicks.grid(column = 3, row = 1, pady = 3, padx = (2,5), sticky = self.E)
            
            if 'angleJetBubble' in self.trackInfo:
                self.labelExtraWidth = tkinter.ttk.Label(self.extraframe, text = 'Particle width')
                self.entryExtraWidth = tkinter.Spinbox(self.extraframe, from_=1, to=180, textvariable = self.extra_partWidth, width = 3)
                self.labelExtraWidth.grid(column = 2, row = 1, pady = 3, padx = (10,0), sticky = self.W)
                self.entryExtraWidth.grid(column = 3, row = 1, pady = 3, padx = (2,5), sticky = self.E)

        ##### Grid configuration #####
        
        self.mainframe.grid(column=0, row=0, sticky = (self.N,self.S,self.E,self.W))
        self.videolbl.grid(column = 0, row = 0, sticky = self.W)
        self.entryVideo.grid(column = 0, row = 1, padx = (3,10), sticky = (self.E,self.W), columnspan = 2 )
        self.entryVideoScroll.grid(column = 0, row = 2, pady = (0,15), padx = (3,10), sticky = (self.E,self.W), columnspan = 2)
        self.videoButton.grid(column = 2, row = 1, sticky = self.E)

        buttonrow = 11
        self.acceptButton.grid(column = 2, row = buttonrow+1, pady = (0,5), sticky = (self.E, self.W))
        self.cancelButton.grid(column = 1, row = buttonrow+1, sticky = self.E, pady = (0,5), padx = (0, 15))
        self.previewButton.grid(column = 2, row = buttonrow, pady = (10,0), sticky = (self.E, self.W))
        self.goBackButton.grid(column = 0, row = buttonrow+1, pady = (5,5), padx = (10,0), sticky = (self.W))

        self.valuebuttonframe.grid(column = 0, row = buttonrow, pady = (5,5), padx = (0,0), sticky = self.W)
        self.defaultButton.grid(column = 0, row = 0, pady = (0,0), padx = (0,0), sticky = (self.W))
        self.previousButton.grid(column = 1, row = 0, pady = (0,0), padx = (0,0), sticky = (self.W))

        #Image treatment options
        self.imgframe.grid(column=0, row = 3, columnspan = 3, sticky = (self.N,self.S,self.E,self.W))
        self.entryMedian.grid(column = 1, row = 0, padx = (3,10), pady = (0,5), sticky = self.W)
        self.entryGaussian.grid(column = 1, row = 1, padx = (3,10), pady = (5,0), sticky = self.W)
        self.labelMedian.grid(column = 0, row = 0, padx = (0,3), pady = (0,5), sticky = self.W)
        self.labelGaussian.grid(column = 0, row = 1, padx = (0,3), pady = (5,0), sticky = self.W)
        
        self.checkEqualizeHist.grid(column = 2, row = 0, columnspan = 2, padx = (3,10), pady = (0,5), sticky = self.W)
        self.entryClahe.grid(column = 3, row = 1, padx = (3,10), pady = (5,0), sticky = self.W)
        self.labelClahe.grid(column = 2, row = 1, padx = (0,3), pady = (5,0), sticky = self.W)

        #68 in padx labelOpen
        self.labelOpen.grid(column = 4, row = 0, pady = (0,5), padx = (10,3), sticky = self.W)
        self.entryOpen.grid(column = 5, row = 0, pady = (0,5), padx = (3,10), sticky = self.W)
        self.labelClose.grid(column = 4, row = 1, pady = (5,0), padx = (10,3), sticky = self.W)
        self.entryClose.grid(column = 5, row = 1, pady = (5,0),  padx = (3,10), sticky = self.W)

        self.bilateralframe.grid(column = 0, row = 2, columnspan = 4, sticky = (self.W, self.E, self.N, self.S))
        self.labelBilateral.grid(column = 0, row = 0, columnspan = 1, pady = (10,0), padx = 0, sticky = (self.W))
        self.entryBilateralD.grid(column = 1, row = 0, pady = (10,0), padx = 5)
        self.entryBilateralColor.grid(column = 2, row = 0, pady = (10,0), padx = 5)
        self.entryBilateralSpace.grid(column = 3, row = 0, pady = (10,0), padx = 5)

        # self.imgframe.columnconfigure(0, weight = 4)
        # self.imgframe.columnconfigure(1, weight = 5)
        # self.imgframe.columnconfigure(2, weight = 4)
        # self.imgframe.columnconfigure(3, weight = 5)

        #Detection and tracking options
        self.trkframe.grid(column = 0, row = 4, columnspan = 3, sticky = (self.N,self.S,self.E,self.W), pady = (10,0))
        self.labelRadi.grid(column = 0, row = 0, sticky = self.W, pady = (0,10), padx = (25,0))
        self.labelMemory.grid(column = 0, row = 1, sticky = self.E, padx = (25,0))
        self.labelMinArea.grid(column = 2, row = 0, sticky = self.W, pady = (0,10))
        self.labelMaxArea.grid(column = 2, row = 1, sticky = self.W)

        self.entryRadi.grid(column = 1, row = 0, padx = (3,40), pady = (0,10))
        self.entryMemory.grid(column = 1, row = 1, sticky = self.W, padx = (3,40))
        self.entryMinArea.grid(column = 3, row = 0, padx = (3,10), pady = (0,10))
        self.entryMaxArea.grid(column = 3, row = 1, padx = (3,10))

        # self.trkframe.columnconfigure(0, weight = 2)
        # self.trkframe.columnconfigure(1, weight = 2)
        # self.trkframe.columnconfigure(2, weight = 2)
        # self.trkframe.columnconfigure(3, weight = 2)

        #Video options
        self.videoframe.grid(column = 0, row = 9, rowspan = 2, columnspan = 3, sticky = (self.N,self.S,self.E,self.W), pady = (10,0))
        self.checkFit.grid(column = 0, row = 0, padx = 10, sticky = self.W)
        self.checkContour.grid(column = 1, row = 0, padx = 10, sticky = self.W)
        self.labelSkip.grid(column = 2, row = 0, padx = (10,0))
        self.entrySkip.grid(column = 3, row = 0, padx = (3,10))
        self.frameVidStart.grid(column = 0, row = 1, sticky = (self.N,self.S,self.E,self.W))
        self.labelVidStart.grid(column = 0, row = 0, padx = (10,3), pady = (10,0), sticky = self.E)
        self.entryVidStart.grid(column = 1, row = 0, padx = (3,10), pady = (10,0), sticky = self.W)
        self.frameVidEnd.grid(column = 1, row = 1, sticky = (self.N,self.S,self.E,self.W))
        self.labelVidEnd.grid(column = 0, row = 0, padx = (10,3), pady = (10,0), sticky = self.E)
        self.entryVidEnd.grid(column = 1, row = 0, padx = (3,10), pady = (10,0), sticky = self.W)
        self.labelFPS.grid(column = 2, row = 1, padx = (10,0), pady = (10,0), sticky = self.W)
        self.entryFPS.grid(column = 3, row = 1, padx = (3,10), pady = (10,0), sticky = self.E)
        self.videoframe.columnconfigure(0, weight = 2)
        self.videoframe.columnconfigure(1, weight = 2)

        #self.sobelframe.columnconfigure(0,weight = 1)
        self.file_opt = self.options = {}
        # [modified] add some format
        self.options['filetypes'] = [('Video files', '.avi'), ('Video files', '.mp4'), ('Video files', '.wmv'), ('All files', '.*')]

        #Main window
        root.title('Tracking')
        root.geometry('+50+50')
        root.protocol("WM_DELETE_WINDOW", lambda : sys.exit())

        #Font text
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def __init_basic__(self):
        #Starts tkinter variables with generic purpose
        self.filename = tkinter.StringVar() #Video filename
        self.vid_fit = tkinter.BooleanVar() #True if the video must fit to screen
        self.vid_skip = tkinter.IntVar() #Makes frame jumps of size 'vid_skip'
        self.vid_start = tkinter.IntVar() #Start the analysis at this frame
        self.vid_end = tkinter.StringVar() #Finish the analysis at this frame
        self.vid_FPS = tkinter.StringVar() #Output frames per second of the video
        self.vid_totalFrames = tkinter.IntVar() #Total amount of frames
        self.vid_lastFilename = tkinter.StringVar() #Last filename used to calculate total frames

        self.default_values = {'filename' : os.getcwd(),
        'vid_fit':True, 
        'vid_skip': 1, 
        'vid_start': 1, 
        'vid_end': -1, 
        'vid_FPS': '',
        'vid_totalFrames': 0,
        'vid_lastFilename':''}
        #Starts non-tkinter variables
        self.loopTrack = True
        
    def create_tkinter_dict(self):
        # Creates dictionary with only tkinter variables accessed by their name
        # [modified]error in py3, can not get BooleanVar & StringVar
        # return {var: self.__dict__[var] for var in self.__dict__
        #           if isinstance(self.__dict__[var], type(tkinter.IntVar()))}
        return {var: self.__dict__[var] for var in self.__dict__ if
                (isinstance(self.__dict__[var], type(tkinter.IntVar()))
                or isinstance(self.__dict__[var], type(tkinter.BooleanVar()))
                or isinstance(self.__dict__[var], type(tkinter.StringVar())))}
   
    def get_filename(self,*args):
        self.options['initialdir'] = self.filename.get()[::-1].partition('/')[2][::-1]  # Gets currently selected path
        self.filename.set(tkinter.filedialog.askopenfilename(**self.file_opt))
        # print(self.filename.get()[::-1].partition('/')[2][::-1])

    def go_back(self, *args):
        # Goes back to the main menu
        self.root.destroy()
        self.loopTrack = False

    def exit_task(self,*args):
        # #Converts local tkinter variables to global
        # for var in self.variables:
        #     globals()[var] = self.variables[var].get()
        self.write_config('Previous_values')
        for var in self.variables:
            self.variables[var] = self.variables[var].get()
            # print(str(var)+":"+str(self.variables[var]))
        self.root.destroy()

    def write_config(self, section, *args):
        # Writes variables to config file
        config = configparser.RawConfigParser()
        config.add_section(section)
        for name in self.variables:
            var = self.variables[name].get()
            if isinstance(var, bool):
                # Boolean variables need to be converted to "int" to be read properly next time
                var = int(var)
            config.set(section, name, var)

        # Recovers the other sections so that it doesn't overwrite them
        configRead = configparser.RawConfigParser()
        configRead.read(self.cfgfilename)
        allSections = configRead.sections()
        allSections = [sec for sec in allSections if sec!=section]
        for sec in allSections:
            config.add_section(sec)
            for option, value in configRead.items(sec):
                config.set(sec, option, value)
        # [modified] error in py3
        # with open(self.cfgfilename, 'wb') as configfile:
        with open(self.cfgfilename, 'w') as configfile:
            config.write(configfile)

    def read_config(self, section, *args):
        # Sets values taken from "section" in the configfile
        config = configparser.RawConfigParser()
        config.read(self.cfgfilename)
        for name in self.variables:
            self.variables[name].set(config.get(section, name))
#            if name == 'filename':
#                self.vid_lastFilename = config.get(section, name)

    def open_config(self, name):
        # Sets the path to the cfg file (either in current directory or in "Files", if it exists)
        # "name" gives the filename
        if os.path.isdir('Files'):
            self.cfgfilename = 'Files/' + name
        else:
            self.cfgfilename = name
            
        # Reads the values of the cfg file, or creates a new one if it does not exist
        try: 
            self.read_config('Previous_values')
        except configparser.NoSectionError:
            self.set_default()
    
    def read_previous_config(self,*args):
        try:
            self.read_config('Previous_values')
        except configparser.NoSectionError:
            pass

    def set_default(self):
        for name in self.variables:
            self.variables[name].set(self.default_values[name])

    def scrollHandler(self, *L):
        op, howMany = L[0], L[1]
        if op == 'scroll':
            units = L[2]
            self.entryVideo.xview_scroll(howMany, units)
        elif op == 'moveto':
            self.entryVideo.xview_moveto(howMany)

    def check_widget(self, var, widget):
        # Enables or disables 'widget' based on the state of 'var'
        if var == True:
            widget.configure(state='disabled')
        else:
            widget.configure(state='normal')
            
    def set_extra_maxticks(self,*args):
        self.extra_maxticks = dfun.get_max_ticks(int(self.extra_partRadi.get()))
        self.entryExtraTicks['values'] = ['MAX'] + [str(i) for i in range(1,self.extra_maxticks)][::-1]

    #### PREVIEW FUNCTIONALITIES ####

    def show_preview(self,*args):

        # Create display window
        if not self.isPreviewOn:
            import detection_auto as det_auto
            # Create tracking class
            self.TrackClass = det_auto.class_factory(self.mode, extraInfo = self.trackInfo[1:])()
#            if 'angleJanus' in self.trackInfo:
#                self.TrackClass.vecDict = dfun.get_vector_circle(ticks = self.variables['extra_ang_ticks'].get(),radi = self.variables['extra_partRadi'].get())

            # Count frames
            self.count_total_frames()

            # Create windows and widgets
            self.preview = tkinter.Toplevel(self.root)
            self.preview.title('Preview')
            self.previewframe = tkinter.ttk.Frame(self.preview, padding = 10)
            self.labelFrame = tkinter.ttk.Label(self.previewframe)
            self.labelBinary = tkinter.ttk.Label(self.previewframe)

            self.numframe = tkinter.ttk.Frame(self.previewframe)
            self.preview_frameNum = tkinter.IntVar()
            self.preview_frameNum.set(1)
            self.preview_framePrev = np.inf 
            self.refreshImg = Image.open('refresh.png').resize((15,15),Image.ANTIALIAS)
            self.refreshImg = ImageTk.PhotoImage(self.refreshImg)
            self.buttonPrevRefresh = tkinter.ttk.Button(self.numframe, image = self.refreshImg, command = self.actualize_preview)
            self.buttonPrevRefresh.image = self.refreshImg

            #Scale
            self.scaleFrameNum = tkinter.Scale(self.numframe, from_=1, to=self.vid_totalFrames.get(), variable = self.preview_frameNum, orient = tkinter.HORIZONTAL)
            self.swidth, self.sheight = dfun.monitor_resolution()

            #Analyze first frame
            tkframe, tkbinary = self.analyze_frame()

            #Put in the images and manage widgets
            self.labelFrame['image'] = tkframe
            self.labelFrame.image = tkframe
            self.labelBinary['image'] = tkbinary
            self.labelBinary.image = tkbinary
            self.previewframe.grid(column = 0, row = 0)
            self.labelFrame.grid(column = 0, row = 0, pady = (0,10))
            self.labelBinary.grid(column = 0, row = 1, pady = (0,0))
            self.numframe.grid(column = 0, row = 2)
            self.scaleFrameNum.grid(column = 0, row = 0, sticky = (self.W, self.E), padx = (0, 10))
            self.buttonPrevRefresh.grid(column = 1, row = 0, sticky = self.E, pady = (10,0))

            #Make the preview image actualize every time a parameter is changed
            for var in self.variables:
                self.variables[var].trace('w',self.actualize_preview)

            self.preview.protocol("WM_DELETE_WINDOW", self.close_preview) #When the user presses the X button
            self.isPreviewOn = True
    
            #Move window to a suitable position
            self.preview.update_idletasks() #Actualize info so that geometry is correct
            rw,_,rx,ry = self.get_geometry(self.root)
            pw,ph,_,_ = self.get_geometry(self.preview)
            # [modified]
            # self.preview.geometry('+' + str(rx + rw) + '+' + str((self.sheight - ph) / 2)) # error in py3
            self.preview.geometry('%sx%s+%s+%s' % (str(pw), str(ph), str(rx+rw+2), str(ry)))

            self.scaleFrameNum.configure(length = pw-50)
            self.preview.bind("<Return>", self.actualize_preview)
            self.preview.bind("<Up>", lambda x: self.preview_arrow(1)) 
            self.preview.bind("<Down>", lambda x: self.preview_arrow(-1))
            self.preview.bind("<Right>", lambda x: self.preview_arrow(1)) 
            self.preview.bind("<Left>", lambda x: self.preview_arrow(-1))
        else:
            self.close_preview()


    def preview_arrow(self, num, *args):
        #Actualizes the frame by summing "num" to the frame number
        self.preview_frameNum.set(self.preview_frameNum.get()+num)
        self.actualize_preview()


    def get_geometry(self,window):
        #Returns the geometry of a window (width, height, x, y) in "int" values
        geom = window.geometry()
        dim = geom.partition('x')
        separate = dim[2].split('+')
        return list(map(int,[dim[0]]+separate))

    def close_preview(self):
        #Handle the closing of the preview window
        self.preview.destroy()
        self.isPreviewOn = False

    def actualize_preview(self, *args):
        #Called when a parameter is called and the preview window is open
        if self.isPreviewOn:
            self.extra_partRadi.get()

            try:

                tkframe, tkbinary = self.analyze_frame()

                self.labelFrame.configure(image = tkframe)
                self.labelFrame.image = tkframe
                self.labelBinary.configure(image = tkbinary)
                self.labelBinary.image = tkbinary

            except ValueError:
                #When the modified value is left blank
                pass

    def analyze_frame(self):

        # TrackClass needs Vars as attribute
        self.TrackClass.Vars = {}
        for var in self.variables:
            self.TrackClass.Vars[var] = self.variables[var].get()

        self.TrackClass.binary_args = []
        self.TrackClass.contourMode = cv2.CHAIN_APPROX_NONE

        if 'angleJanus' in self.trackInfo:
            self.TrackClass.vecDict = dfun.get_vector_circle(radi=self.TrackClass.Vars['extra_partRadi'], ticks=self.TrackClass.Vars['extra_ang_ticks'])

        if self.mode == 'bgextr':
            # Get reference image
            if self.bg_refOK['Total Average'] == True:
                self.TrackClass.bg_reference = self.TrackClass.smoothing(self.bg_reference[1])[1]
            else:
                self.TrackClass.bg_reference = self.TrackClass.smoothing(self.get_reference())[1]
            self.TrackClass.bg_mode = self.bg_mode.get()
            self.TrackClass.binary_args = self.TrackClass.bg_reference


        label = 0

        frame = self.catch_frame()
        self.TrackClass.centerDict = {}
        # Treat image and get contours
        frame, frame_gray, binary, contours = self.TrackClass.treat_image(frame, self.TrackClass.binary_args) 

        binary_show = binary.copy(); pheight, pwidth = binary.shape
        M_list, cnt_list = self.TrackClass.get_contours(contours)           
                    
        for Mind, M in enumerate(M_list):
            label += 1
            center = tuple(self.TrackClass.get_contour_information(M, frame_gray = frame_gray, cnt = cnt_list[Mind], binary = binary))
            data = [center] + [self.TrackClass.centerDict[center, info] for info in self.trackInfo[1:]]  # Particle info
            self.TrackClass.draw_frame(frame, label, data)

        # Write frame
        cv2.putText(frame, str(self.count) + '/' + str(self.vid_totalFrames.get()), (5, 15), self.font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        # The image needs to fit verticaly in a little less than half of the screen
        needsFitting, ff = dfun.needs_fitting(pwidth, pheight, 0.95*self.swidth, 0.45*self.sheight-25) 

        if needsFitting:
            frame = cv2.resize(frame, None, fx=ff, fy=ff, interpolation=cv2.INTER_LINEAR)
            binary_show = cv2.resize(binary_show, None, fx=ff, fy=ff, interpolation=cv2.INTER_LINEAR)

        # Convert result to img type readable by Tkinter
        tkframe = TKimg_from_array(frame)

        tkbinary = TKimg_from_array(binary_show)
        return tkframe, tkbinary

    def catch_frame(self):
        # Get the frame based on the current position of the self.vid iterator
        frameNum = self.preview_frameNum.get()
        
        if frameNum != self.preview_framePrev:

            if self.isCv2Accurate:
                #Just move to the selected frame if OpenCv is accurate
                self.count = frameNum
                self.vid.set(1, frameNum-1)
                ret, self.frame = self.vid.read()

            else:
                #Seek frame using VideoCapture.read()
                if self.preview_framePrev > frameNum:
                    #If we need a frame previous to the current one, load video from start
                    self.vid = cv2.VideoCapture(self.variables['filename'].get())
                    self.count = 0

                #Run video until chosen frame is reached
                while self.count != frameNum:
                    self.count += 1
                    ret, throwframe = self.vid.read()
                    if ret == False:
                        #If the end is reached
                        break

                self.frame = throwframe

                self.preview_framePrev = self.count

        return self.frame

    def count_total_frames(self, *args):
        #Runs the whole video and stores the total amount of frames (Open CV is not so reliable)
        filename = self.variables['filename'].get()
        if self.vid_lastFilename.get() != filename or self.vid_totalFrames.get() == 0:
            #Frames are manually counted when:
            # - Filename has changed since the last time everything was counted
            # - vid_totalFrames is empty (equals 0)
            # - mode is bgextr and the reference is not properly saved
            vid = cv2.VideoCapture(filename)
            count = 0
            tt = time.time()
            while True:
                # [modified] seem no effect, delete
                # if count in [1]:
                    #Opening a window after vid.read() makes a "segmentation fault" error not appear with weird codecs
                    # cv2.namedWindow('temp')
                    # cv2.imshow('temp', np.ones((1,1,1)))
                    # cv2.waitKey(1)
                    # cv2.destroyAllWindows()
                    # cv2.imshow('temp', np.ones((1,1,1)))

                ret, frame = vid.read()
                if ret == False:
                    break

                count += 1

            self.vid_totalFrames.set(count)
            self.vid_lastFilename.set(filename)

            print('Time to preview was:', time.time()-tt)
            print('Counted frames:', self.vid_totalFrames.get())
            print('Counted frames by OpenCV:', int(vid.get(7))) #Number of frames

            self.isCv2Accurate = self.vid_totalFrames.get() == int(vid.get(7)) #Can OpenCV compute things like FPS, video position,... accurately?
            self.vid = cv2.VideoCapture(filename)
            
        elif self.vid_lastFilename.get() == filename:
            vid = cv2.VideoCapture(filename)            
            print('Counted frames:', self.vid_totalFrames.get())
            print('Counted frames by OpenCV:', int(vid.get(7))) #Number of frames            

    #Calculate FPS window

    def calculate_FPS_window(self, window, *args):
        self.calculateFPS = tkinter.Toplevel(self.root)
        self.calculateFPS.grab_set() #Force focus on this window
        self.calculateFPS.title('Calculate FPS')

        #Variables
        self.fps_seconds = tkinter.DoubleVar() #Total seconds of the video
        self.fps_frames = tkinter.IntVar() #Total frames of the video
        self.fps_fps = tkinter.DoubleVar() #Frames per second of the video      
        self.fps_window = window #"data" if called from post-processing, "vid" if called from main window
        
        if self.fps_window == 'data':
            self.fps_frames.set(self.Vars['totalFrames'])
            self.fps_filename = self.Vars['filename']
        elif self.fps_window == 'vid':
            self.fps_frames.set(self.vid_totalFrames.get())
            self.fps_filename = self.filename.get()
        
        self.fps_seconds.set(0)

        #Widgets
        self.fpsframe = tkinter.ttk.Frame(self.calculateFPS, padding = 10)
        self.fps_labelCalculateFPS = tkinter.ttk.Label(self.fpsframe, text = 'Fill the following information to compute fps:')
        self.fps_labelSeconds = tkinter.ttk.Label(self.fpsframe, text = 'Seconds:')
        self.fps_entrySeconds = tkinter.ttk.Entry(self.fpsframe, textvariable = self.fps_seconds, width = 5)
        self.fps_labelFrames = tkinter.ttk.Label(self.fpsframe, text = 'Frames:')
        self.fps_entryFrames = tkinter.ttk.Entry(self.fpsframe, textvariable = self.fps_frames, width = 5)
        self.fps_labelFPS = tkinter.ttk.Label(self.fpsframe, text = 'FPS:')
        self.fps_labelFPS2 = tkinter.ttk.Label(self.fpsframe, textvariable = self.fps_fps)
        self.fps_buttonCancel = tkinter.ttk.Button(self.fpsframe, text = 'Cancel', width = 4, command = lambda :self.fps_exit(False))
        self.fps_buttonAccept = tkinter.ttk.Button(self.fpsframe, text = 'Accept', width = 7, command = lambda :self.fps_exit(True))

        self.fps_computeFrames_img = Image.open('calculate_frames.jpg').resize((15,15),Image.ANTIALIAS)
        self.fps_computeFrames_img = ImageTk.PhotoImage(self.fps_computeFrames_img)
        self.fps_computeFrames = tkinter.ttk.Button(self.fpsframe, image = self.fps_computeFrames_img, command = self.fps_compute_frames)
        self.fps_computeFrames.image = self.fps_computeFrames_img

        #Grid
        self.fpsframe.grid(column = 0, row = 0)
        self.fps_labelCalculateFPS.grid(column = 0, row = 0, columnspan = 5, padx = 5, pady = 5, sticky = self.W)
        self.fps_labelSeconds.grid(column = 0, row = 1, padx = (5,0), pady = 5, sticky = self.W)
        self.fps_entrySeconds.grid(column = 1, row = 1, padx = (2,5), pady = 5, sticky = self.W)
        self.fps_labelFrames.grid(column = 2, row = 1, padx = (5,0), pady = 5, sticky = self.W)
        self.fps_entryFrames.grid(column = 3, row = 1, padx = (2,0), pady = 5, sticky = self.W)
        self.fps_computeFrames.grid(column = 4, row = 1, padx = (0,5), pady = 5, sticky = self.W)
        self.fps_labelFPS.grid(column = 0, row = 2, padx = (5,0), pady = 5, sticky = self.W)
        self.fps_labelFPS2.grid(column = 1, row = 2, columnspan = 3, padx = (0,5), pady = 5, sticky = self.W)
        self.fps_buttonCancel.grid(column = 0, row = 3, columnspan = 1, padx = 5, pady = 5, sticky = (self.W, self.E))
        self.fps_buttonAccept.grid(column = 3, row = 3, columnspan = 2, padx = 5, pady = 5, sticky = (self.W, self.E))

        self.fpsframe.columnconfigure(0, weight = 1)
        self.fpsframe.columnconfigure(1, weight = 1)
        self.fpsframe.columnconfigure(2, weight = 1)
        self.fpsframe.columnconfigure(3, weight = 1)
        self.fpsframe.columnconfigure(4, weight = 1)

        #When the X button is pressed
        self.calculateFPS.protocol("WM_DELETE_WINDOW", lambda : self.fps_exit(False)) 
        
        #Trace variables
        self.fps_seconds.trace('w',self.fps_compute)
        self.fps_frames.trace('w',self.fps_compute)
        self.fps_compute()

    def fps_compute(self, *args):
        #Computes FPS if the seconds and frames are given
        try:
            frames, secs = self.fps_frames.get(), self.fps_seconds.get()
            if frames and secs:
                self.fps_fps.set(frames/float(secs))

        except ValueError:
            #When get() returns empty value
            pass
            


    def fps_exit(self, accept, *args):
        #Close the window and return focus to main. If "accept" button was clicked, save the data.
        fps = self.fps_fps.get()
        if accept:
            if fps:
                if self.fps_window == 'data':
                    self.data_FPS.set(str(fps))
                if self.fps_window == 'vid':
                    self.vid_FPS.set(str(fps))
            
        self.calculateFPS.grab_release()
        self.calculateFPS.destroy()

    def fps_compute_frames(self, *args):
        vid = cv2.VideoCapture(self.fps_filename)
        count = 0

        while True:
            ret, _ = vid.read()
            if ret == False:
                break
            count += 1

        self.fps_frames.set(count)
        if self.fps_window == 'data':
            self.Vars['totalFrames'] = count
        elif self.fps_window == 'vid':
            self.vid_totalFrames.set(count)
            self.vid_lastFilename.set(self.fps_filename)







class dataAnalysisGUI(mainGUI):
    
    def __init__(self, root, Vars = {}, infoDict = {}):

        #Variables
        self.data_ParticleList = tkinter.StringVar() #List of particles that will be analyzed
        self.data_checkAll = tkinter.BooleanVar() #True if the user wants all particles analyzed
        self.data_draw = tkinter.BooleanVar() #If True the trajectories of the selected particles will be drawn
        self.color_B = tkinter.IntVar() #Blue color for the trajectory drawing
        self.color_G = tkinter.IntVar() #Green color for the trajectory drawing
        self.color_R = tkinter.IntVar() #Red color for the trajectory drawing
        self.color_rand = tkinter.BooleanVar() #Randomize trajectory color
        self.color_randNum = tkinter.IntVar() #Number of randomized colors
        self.color_thick = tkinter.IntVar() #Thickness of the trajectory line
        self.color_label = tkinter.BooleanVar() #Show particle label in video
        self.color_orientation = tkinter.BooleanVar() #Show particle orientation in video
        self.color_orientation_interval = tkinter.IntVar() #Interval of frames between each drawing of orientation
        self.data_FPS = tkinter.StringVar() #Frames per second
        self.data_pix2mtr = tkinter.StringVar() #Meters per pixel
        self.data_interp = tkinter.StringVar() #Mode of interpolation
        self.data_order = tkinter.IntVar() #Order of centered derivative
        self.data_anglevel = tkinter.BooleanVar() #Wether or not to calculate angle from velocity
        self.data_AAC = tkinter.BooleanVar() #Wether or not to calculate the angular auto-correlation
        # [modified]
        self.color_speed = tkinter.BooleanVar()  # Color change with speed
        self.min_speed = tkinter.StringVar()  # Minimun speed color is blue
        self.max_speed = tkinter.StringVar()  # Maximum speed color is red
        # [modified end]

        self.variables = { var: self.__dict__[var] for var in self.__dict__} #Store only the Tkinter variables

        self.Vars = Vars
        self.infoDict = infoDict
        self.arrayColorImage = np.zeros((20,100,3), dtype = np.uint8) #Image to show the color
        self.loopProcessing = True #To go back to the previous menu
        
        self.default_values = {
            'data_ParticleList': '',
            'data_checkAll': True,
            'data_draw': True,
            'color_B': 255,
            'color_G':0,
            'color_R':0,
            'color_rand':True,
            'color_randNum':5,
            'color_thick': 1,
            'color_label' : True,
            'color_orientation': True,
            'color_orientation_interval': 0,
            'data_FPS': '',
            'data_pix2mtr': '',
            'data_interp':'cubic',
            'data_order':2,
            'data_anglevel': False,
            'data_AAC': False,
            # [modified]
            'color_speed': False,
            'min_speed': 2,
            'max_speed': 10
            # [modified end]
        }


        self.open_config('GUI_data_config.cfg')

        #Corrector variables
        self.correct_delParticle = tkinter.IntVar() #Currently selected particle to delete frames
        self.correct_delFrames = tkinter.StringVar() #Currently selected list of frames to be deleted
        self.correct_delStart = tkinter.StringVar() #Starting frame of the currently selected particle to correct
        self.correct_delEnd = tkinter.StringVar() #Last frame of the currently selected particle to correct
        self.correct_deleteDict = {p:'' for p in self.get_particle_list()} #{ Particle to be corrected: [frames to be deleted]}
        self.correct_delParticle.trace('w',self.update_delFrameList)
        self.correct_delFrames.trace('w',self.update_delDict)
        
        self.correct_joinParticle1 = tkinter.IntVar() #First particle to join
        self.correct_joinParticle2 = tkinter.StringVar() #Second particle to join
        self.correct_joinDict = {p:'None' for p in self.get_particle_list()} #{particle1: particle2 to join with}
        self.correct_joinParticle1.trace('w', self.update_joinParticle2)
        self.correct_joinParticle2.trace('w', self.update_joinDict)
        
        self.correct_startend = self.get_startend_dic(self.infoDict, self.get_particle_list()) #{ particle : (start,end)}
        self.correct_startend_original = copy.deepcopy(self.correct_startend)

        
        #Main variables and bindings
        root.bind('<Return>', self.exit_data_task) #Binds enter to the start button
        root.bind('<Escape>', lambda *args:sys.exit()) #Binds ESC to the exit button
        self.N, self.S, self.E, self.W = tkinter.N, tkinter.S, tkinter.E, tkinter.W

        #Main buttons
        self.root = root
        root.title('Post-processing options')
        root.geometry('+100+100')
        self.mainframe = tkinter.ttk.Frame(root, padding = (10,10,10,10))
        self.acceptButton = tkinter.ttk.Button(self.mainframe, text = 'Continue', command = self.exit_data_task)
        self.acceptButton.state(['active'])
        self.cancelButton = tkinter.ttk.Button(self.mainframe, text = 'Cancel', command = lambda : sys.exit())
        self.repeatButton = tkinter.ttk.Button(self.mainframe, text = 'Repeat', command = self.repeat)

        #Entering data
        self.labelTitle = tkinter.ttk.Label(self.mainframe, text = 'Which particles you want to analyze?')
        self.entryParticles = tkinter.ttk.Entry(self.mainframe, textvariable = self.data_ParticleList) #Entry
        self.checkAll = tkinter.ttk.Checkbutton(self.mainframe, text = 'All', variable = self.data_checkAll, command = lambda: self.check_widget(self.data_checkAll.get(),self.entryParticles), onvalue = True, offvalue = False) #Entry is deactivated when "all" is selected


        ##### Drawing options #####
        self.drawframe = tkinter.ttk.Labelframe(self.mainframe, text = ' Drawing options: ', padding = 10, relief = 'sunken')
        self.checkDraw = tkinter.ttk.Checkbutton(self.drawframe, text = 'Draw trajectory', variable = self.data_draw, onvalue = True, offvalue = False, command = self.check_colors)
        self.labelColor = tkinter.ttk.Label(self.drawframe, text = 'Color:')

        self.colorframe = tkinter.ttk.Frame(self.drawframe) #, padding = 10)

        self.labelColorB = tkinter.ttk.Label(self.colorframe, text = 'B')
        self.labelColorG = tkinter.ttk.Label(self.colorframe, text = 'G')
        self.labelColorR = tkinter.ttk.Label(self.colorframe, text = 'R')
        self.entryColorB = tkinter.Spinbox(self.colorframe, from_=0, to=255, textvariable = self.color_B, width = 3)
        self.entryColorG = tkinter.Spinbox(self.colorframe, from_=0, to=255, textvariable = self.color_G, width = 3)
        self.entryColorR = tkinter.Spinbox(self.colorframe, from_=0, to=255, textvariable = self.color_R, width = 3)
        self.labelColorImage = tkinter.ttk.Label(self.drawframe) #Create label for color image

        self.color_B.trace('w',self.update_color_image)
        self.color_G.trace('w',self.update_color_image)
        self.color_R.trace('w',self.update_color_image)

        #Randomize
        self.checkRand = tkinter.ttk.Checkbutton(self.drawframe, text = 'Randomize', variable = self.color_rand, onvalue = True, offvalue = False, command = self.update_color_image)
        self.randnumframe = tkinter.ttk.Frame(self.drawframe)
        self.labelRandNum = tkinter.ttk.Label(self.randnumframe, text = '#')
        self.entryRandNum = tkinter.Spinbox(self.randnumframe, from_=1, to=255, textvariable = self.color_randNum, width = 2)
        
        self.color_randNum.trace('w',self.update_color_image)

        self.update_color_image() #Start the color image
        self.labelColorImage.image = self.colorImage #Attach the image
        self.labelColorImage.bind('<Button-1>', self.update_color_image)
        # [modified]
        self.colorSpeedFrame = tkinter.ttk.Frame(self.drawframe)
        self.checkColorSpeed = tkinter.ttk.Checkbutton(self.colorSpeedFrame, text='Color change with speed', variable=self.color_speed, onvalue=True, offvalue=False, command=self.colorspeed_check)
        self.labelMinSpeed = tkinter.ttk.Label(self.colorSpeedFrame, text='Minimum speed')
        self.labelMaxSpeed = tkinter.ttk.Label(self.colorSpeedFrame, text='Maximum speed')
        self.entryMinSpeed = tkinter.ttk.Entry(self.colorSpeedFrame, textvariable=self.min_speed, width=3)
        self.entryMaxSpeed = tkinter.ttk.Entry(self.colorSpeedFrame, textvariable=self.max_speed, width=3)
        # [modified end]

        #Thickness and label
        self.thickframe = tkinter.ttk.Frame(self.drawframe)
        self.labelThick = tkinter.ttk.Label(self.thickframe, text = 'Thickness: ')
        self.entryThick = tkinter.Spinbox(self.thickframe, from_=1, to=1000, textvariable = self.color_thick, width = 2)
        self.checkLabel = tkinter.ttk.Checkbutton(self.drawframe, text = 'Show particle label', variable = self.color_label, onvalue = True, offvalue = False)
        
        #Draw orientation
        self.orientationframe = tkinter.ttk.Frame(self.drawframe)
        self.checkOrientation = tkinter.ttk.Checkbutton(self.orientationframe, text='Show orientation every', variable = self.color_orientation, onvalue = True, offvalue = False, command = self.check_orientation_widgets)
        self.entryOrientationInterval = tkinter.ttk.Entry(self.orientationframe, textvariable = self.color_orientation_interval, width = 3)
        self.labelOrientationInterval = tkinter.ttk.Label(self.orientationframe, text='frames')

        self.check_widget(self.data_checkAll.get(),self.entryParticles) #Check 
        self.check_colors()
        self.check_orientation_widgets()


        ##### Interpolation and finite differences #####
        self.dataframe = tkinter.ttk.Labelframe(self.mainframe, text = ' Data processing: ', padding = 10, relief = 'sunken')

        #Entering unit conversion
        self.unitframe = tkinter.ttk.Frame(self.dataframe) #, padding = (5,10))
        self.labelUnit = tkinter.ttk.Label(self.dataframe, text = 'Unit conversion:')
        self.entryFPS = tkinter.ttk.Entry(self.unitframe, textvariable = self.data_FPS, width = 4)
        self.buttonFPS = tkinter.ttk.Button(self.unitframe, width = 4, text = 'FPS', command = lambda :self.calculate_FPS_window('data'))
        self.cancelButton = tkinter.ttk.Button(self.mainframe, text = 'Cancel', command = lambda : sys.exit())

        self.entryPix2mtr = tkinter.ttk.Entry(self.unitframe, textvariable = self.data_pix2mtr, width = 4)
        self.labelPix2mtr = tkinter.ttk.Label(self.unitframe, text = 'Pixels per \u03BC'+ 'm')

        self.labelInterp = tkinter.ttk.Label(self.dataframe, text = 'Interpolation method for missing points')
        self.entryInterp = tkinter.ttk.Combobox(self.dataframe, textvariable = self.data_interp, width = 10)
        self.entryInterp['values'] = ('linear', 'cubic', 'slinear', 'quadratic', 'nearest', 'zero')
        self.entryInterp.state(['readonly'])
        self.labelOrder = tkinter.ttk.Label(self.dataframe, text = 'Velocity precision')
        self.entryOrder = tkinter.Spinbox(self.dataframe, from_=2, to=8, increment_=2, width = 4, textvariable = self.data_order)
        
        self.checkAnglevel = tkinter.ttk.Checkbutton(self.dataframe, text = 'Compute angle from instantaneous velocity', variable = self.data_anglevel, onvalue = True, offvalue = False)
        self.checkAAC = tkinter.ttk.Checkbutton(self.dataframe, text = 'Compute angular auto-correlation', variable = self.data_AAC, onvalue = True, offvalue = False)
       
        #Grid configuration
        self.mainframe.grid(column = 0, row = 0, sticky = (self.N,self.S,self.E,self.W))
        self.labelTitle.grid(column = 0, row = 0, columnspan = 2, sticky = (self.E, self.W), pady = (5,10))
        self.checkAll.grid(column = 0, row = 1, sticky = self.W, padx = (5,5))
        self.entryParticles.grid(column = 1, row = 1, sticky = (self.W, self.E), padx = (5,10))

        self.repeatButton.grid(column = 0, row = 6, sticky = self.W, padx = 5, pady = (5,0))
        self.cancelButton.grid(column = 0, row = 7, sticky = self.W, padx = 5, pady = (3,0))
        self.acceptButton.grid(column = 1, row = 7, sticky = self.E, padx = 5, pady = (3,0))

        #Drawing options configuration
        self.drawframe.grid(column = 0, row = 2, columnspan = 2, pady = 3, sticky = (self.W, self.E, self.N, self.S))
        self.checkDraw.grid(column = 0, row = 0, columnspan = 2, sticky = (self.W, self.E, self.S), padx = (5,10), pady = 5)
        self.thickframe.grid(column = 1, row = 0, sticky = self.W, padx = 5, pady = 5)

        self.labelColor.grid(column = 0, row = 1, sticky = (self.W, self.E), padx = (5,5), pady = (5,5))
        self.colorframe.grid(column = 1, row = 1, columnspan = 2, sticky = (self.W, self.E), padx = (5,10), pady = (5,5))
        self.labelColorB.grid(column = 0, row = 0, sticky = self.E, padx = (5,0))
        self.entryColorB.grid(column = 1, row = 0, sticky = self.W, padx = (0,5))
        self.labelColorG.grid(column = 2, row = 0, sticky = self.E, padx = (5,0))
        self.entryColorG.grid(column = 3, row = 0, sticky = self.W, padx = (0,5))
        self.labelColorR.grid(column = 4, row = 0, sticky = self.E, padx = (5,0))
        self.entryColorR.grid(column = 5, row = 0, sticky = self.W, padx = (0,5))
        self.labelColorImage.grid(column = 3, row = 1)

        self.checkRand.grid(column = 0, row = 2, sticky = self.W, padx = (5,0), pady = 5)
        self.randnumframe.grid(column = 3, row = 2, sticky = self.E, padx = 5, pady = 5)
        self.labelRandNum.grid(column = 0, row = 0, sticky = self.W)
        self.entryRandNum.grid(column = 1, row = 0, sticky = self.W)

        # [modified]
        self.colorSpeedFrame.grid(column=0, row=4, columnspan=2, pady=3, sticky=(self.W, self.E))
        self.checkColorSpeed.grid(column=0, row=0, sticky=self.W, padx=(5, 0), pady=5)
        self.entryMaxSpeed.grid(column=4, row=0, sticky=self.E)
        self.labelMaxSpeed.grid(column=3, row=0, sticky=self.E, padx=(10, 0))
        self.entryMinSpeed.grid(column=2, row=0, sticky=self.E)
        self.labelMinSpeed.grid(column=1, row=0, sticky=self.E, padx=(10, 0))
        # [modified end]

        self.checkLabel.grid(column = 0, row = 3, sticky = self.W, padx = (5,0), pady = (5,2))
        self.labelThick.grid(column = 0, row = 3, sticky = self.W, padx = (10,0), pady = 1)
        self.entryThick.grid(column = 1, row = 3, sticky = self.W, padx = (1,1), pady = 1)
        
        self.orientationframe.grid(column = 1, row = 3, columnspan = 3, sticky = (self.W, self.E), padx = (10,5), pady = (5,2))
        self.checkOrientation.grid(column = 0, row = 0, sticky = self.W, padx = (5,3), pady = (0,0))
        self.entryOrientationInterval.grid(column = 1, row = 0, sticky = self.W, padx = (0,0), pady = (0,0))
        self.labelOrientationInterval.grid(column = 2, row = 0, sticky = self.W, padx = (3,0), pady = (0,0))


        # self.drawframe.columnconfigure(1, weight = 1)
        # self.drawframe.columnconfigure(2, weight = 100)


        #Data options configuration
        #Unit conversion configuration
        self.dataframe.grid(column = 0, row = 4, columnspan = 2, padx = 0, pady = 3, sticky = (self.W, self.E, self.N, self.S))

        self.unitframe.grid(column = 1, row = 0, columnspan = 2, padx = 5, pady = (0,5), sticky = (self.W, self.E, self.N, self.S))
        self.labelUnit.grid(column = 0, row = 0, padx = 5, pady = (0,5), sticky = self.W)
        self.entryFPS.grid(column = 0, row = 0, padx = (0,2), pady = 0, sticky = self.W)
        #self.labelFPS.grid(column = 1, row = 0, padx = (2,10), pady = 0, sticky = self.W)
        self.buttonFPS.grid(column = 1, row = 0, padx = (2,10), pady = 0, sticky = self.W)
        self.entryPix2mtr.grid(column = 2, row = 0, padx = (10,2), pady = 0, sticky = self.W)
        self.labelPix2mtr.grid(column = 3, row = 0, padx = (2,10), pady = 0, sticky = self.W)

        #Interpolation configuration
        self.labelInterp.grid(column = 0, row = 1, columnspan = 2, padx = (5,5), pady = (7,5), sticky = self.W)
        self.entryInterp.grid(column = 2, row = 1, pady = (5,5), padx = (5,5), sticky = self.W)
        self.labelOrder.grid(column = 0, row = 2, pady = (5,5), padx = (5,5), sticky = self.W)
        self.entryOrder.grid(column = 1, row = 2, pady = (5,5), padx = (5,5), sticky = self.W)
        
        self.checkAnglevel.grid(columnspan = 2, row = 3, pady = (5,5), padx = 5, sticky = self.W)        
        self.checkAAC.grid(columnspan = 2, row = 4, pady = (5,5), padx = 5, sticky = self.W)
    
        ##### CORRECTING OPTIONS #####
        self.correctorframe = tkinter.ttk.Labelframe(self.mainframe, text = ' Correcting options: ', padding = 10, relief = 'sunken')
        
        self.deleteframe = tkinter.ttk.Frame(self.correctorframe)
        self.labelDelete = tkinter.ttk.Label(self.deleteframe, text = 'Delete points: ')
        self.labelDeleteParticle = tkinter.ttk.Label(self.deleteframe, text = 'Particle')
        self.entryDeleteParticle = tkinter.ttk.Combobox(self.deleteframe, textvariable = self.correct_delParticle, width = 3, height = 10, postcommand = self.update_delParticleList)
        self.labelDeleteFrame = tkinter.ttk.Label(self.deleteframe, text = ', frames ')
        self.entryDeleteFrame = tkinter.ttk.Entry(self.deleteframe, textvariable = self.correct_delFrames, width = 10)
        
        self.labelDeleteStart = tkinter.ttk.Label(self.deleteframe, text = 'Start:')
        self.labelDeleteStart2 = tkinter.ttk.Label(self.deleteframe, textvariable = self.correct_delStart)
        self.labelDeleteEnd = tkinter.ttk.Label(self.deleteframe, text = 'End:')
        self.labelDeleteEnd2 = tkinter.ttk.Label(self.deleteframe, textvariable = self.correct_delEnd)
        
        self.joinframe = tkinter.ttk.Frame(self.correctorframe)
        self.labelJoin = tkinter.ttk.Label(self.joinframe, text = 'Join trajectories: ')
        self.labelJoinParticle1 = tkinter.ttk.Label(self.joinframe, text = 'Particle ')
        self.entryJoinParticle1 = tkinter.ttk.Combobox(self.joinframe, textvariable = self.correct_joinParticle1, width = 3, height = 10, postcommand = self.update_delParticleList)
        self.labelJoinParticle2 = tkinter.ttk.Label(self.joinframe, text = ' with ')
        self.entryJoinParticle2 = tkinter.ttk.Combobox(self.joinframe, textvariable = self.correct_joinParticle2, width = 3, height = 10, postcommand = self.update_joinParticle2List)
        

        #Grid        
        self.correctorframe.grid(column = 0, row = 5, columnspan = 2, padx = 0, pady = 3, sticky = (self.W, self.E, self.N, self.S))

        self.deleteframe.grid(column = 0, row = 0, columnspan = 2)
        self.labelDelete.grid(column = 0, row = 0, padx = 5, pady = 5, sticky = self.W)
        self.labelDeleteParticle.grid(column = 1, row = 0, padx = 0, pady = 5, sticky = self.E)
        self.entryDeleteParticle.grid(column = 2, row = 0, padx = 0, pady = 5, sticky = self.W)
        self.labelDeleteFrame.grid(column = 3, row = 0, padx = 0, pady = 5, sticky = self.W)
        self.entryDeleteFrame.grid(column = 4, row = 0, padx = 0, pady = 5, sticky = self.W)
        
        self.labelDeleteStart.grid(column = 5, row = 0, padx = (10,0), pady = 5, sticky = self.W)
        self.labelDeleteStart2.grid(column = 6, row = 0, padx = (0,0), pady = 5, sticky = self.W)
        self.labelDeleteEnd.grid(column = 7, row = 0, padx = (5,0), pady = 5, sticky = self.W)
        self.labelDeleteEnd2.grid(column = 8, row = 0, padx = (0,0), pady = 5, sticky = self.W)
        
        self.joinframe.grid(column = 0, row = 1, columnspan = 2, sticky = self.W)
        self.labelJoin.grid(column = 0, row = 0, padx = 5, pady = 5, sticky = self.W)
        self.labelJoinParticle1.grid(column = 1, row = 0, padx = (0,0), pady = 5, sticky = self.W)
        self.entryJoinParticle1.grid(column = 2, row = 0, padx = (0,0), pady = 5, sticky = self.W)
        self.labelJoinParticle2.grid(column = 3, row = 0, padx = (0,0), pady = 5, sticky = self.W)
        self.entryJoinParticle2.grid(column = 4, row = 0, padx = (0,0), pady = 5, sticky = self.W)

        
        self.mainframe.columnconfigure(0, weight = 1)
        self.mainframe.columnconfigure(1, weight = 1)
        self.mainframe.columnconfigure(2, weight = 1)
        self.mainframe.columnconfigure(3, weight = 1)
        self.mainframe.columnconfigure(4, weight = 1)

        root.protocol("WM_DELETE_WINDOW", lambda : sys.exit())


       
    def exit_data_task(self,*args):  
        # Converts local tkinter variables to global
        self.write_config('Previous_values')

        if self.data_checkAll.get() == True:
            # If "all" is selected ignore the current list
            self.data_ParticleList.set('')
            
        self.exit_data_corrector() # Treats variables related to corrector

        for var in self.variables:
            globals()[var] = self.variables[var].get()

        self.root.destroy()


    def repeat(self, *args):
        # Exit GUI and break post-processing loop
        self.root.destroy()
        self.loopProcessing = False

    def colorspeed_check(self, *args):

        pass

    def update_color_image(self, *args):
        # Updates the color image
        if not self.color_rand.get():
            self.update_color_image_single()

        elif len(args) == 0 or args[0] not in ('PY_VAR25', 'PY_VAR26', 'PY_VAR27'):
            self.update_color_image_rand()

        # Move array image to label image
        self.colorImage = TKimg_from_array(self.arrayColorImage)
        self.labelColorImage.configure(image = self.colorImage)
        self.labelColorImage.image = self.colorImage
        
    def update_color_image_single(self, *args):
        # Update color image with single color
        try:
            B, G, R = self.color_B.get(), self.color_G.get(), self.color_R.get()
            # We paint the color image taking into account that tkinter is RGB instead of BGR
            self.arrayColorImage[:, :, 0] = B
            self.arrayColorImage[:, :, 1] = G
            self.arrayColorImage[:, :, 2] = R
            self.colorList = [[B, G, R]]
        # [modified] ValueError can not be caught in py3
        # except ValueError:
        except tkinter.TclError:
            # When get() returns an empty value
            return

    def update_color_image_rand(self, *args):
        # Randomize color!
        try:
            number = self.color_randNum.get()
        # except ValueError:
        except tkinter.TclError:
            # When get() returns an empty value
            return
        self.colorList = dfun.get_rand_colors(number)

        colorImageWidth = self.arrayColorImage.shape[1]
        # [modified] error in py3 (or numpy 1.12), slice index should be integer
        # colorWidth = colorImageWidth / number
        colorWidth = colorImageWidth//number #Length occupied by each random color
        for i in range(number):
            bgr = self.colorList[i]
            self.arrayColorImage[:, i * colorWidth: (i + 1) * colorWidth] = (bgr[2], bgr[1], bgr[0])

        if colorImageWidth % number != 0:
            # Not all the image will be covered by the bins: stretch it!
            resizeWidth = colorImageWidth / float(number * colorWidth)
            self.arrayColorImage = cv2.resize(self.arrayColorImage[:, :number*colorWidth, :], None,
                                              fx=resizeWidth, fy=1, interpolation=cv2.INTER_LINEAR)

        self.colorImage = TKimg_from_array(self.arrayColorImage)
        self.labelColorImage.configure(image = self.colorImage)
        self.labelColorImage.image = self.colorImage

    def check_orientation_widgets(self, *args):
        # Disables orientation interval depending on the check button "draw orientation"
        self.check_widget(not self.color_orientation.get(), self.entryOrientationInterval)
        self.check_widget(not self.color_orientation.get(), self.labelOrientationInterval)

    def check_colors(self, *args):
        # Disables all drawing options depending on the variable "data_draw". Iterates over frames
        boolean = self.data_draw.get()
        
        childlistContainer = [[child for child in self.drawframe.winfo_children() if child != self.checkDraw]]
        while len(childlistContainer) > 0:
            children = childlistContainer.pop(0)
            for widget in children:
                if widget.winfo_class() != 'TFrame':
                    self.check_widget(not boolean, widget) #Change status depending on boolean
                else:
                    childlistContainer.append(widget.winfo_children()) #If it's a frame widget, add the children widgets to the list for the next loop

    def update_delDict(self, *args):
        #Update delete dictionary based on the variables
        p, l = self.correct_delParticle.get(), self.correct_delFrames.get()
        self.correct_deleteDict[p] = l
        #Did it change the starting/ending point of the particle?
        pos = copy.deepcopy(self.infoDict[p, 'position'])
        l = dfun.interpret_list(l)
        if len(l) > 0: #non-empty list
            start, end = self.correct_startend_original[p]
            for f in l:
                if start <= f <= end:
                    f = (f - self.Vars['vid_start'])/self.Vars['vid_skip']
                    pos[f] = (None,None)

            startend = self.get_startend_dic({(p,'position'):pos}, [p])
            self.correct_startend[p] = startend[p] #Actualize start/end
            self.update_delStartEndLabel(p)

    def update_delFrameList(self, *args):
        #Sets appropriate frame list for deletion
        particle = self.correct_delParticle.get()
        self.correct_delFrames.set(self.correct_deleteDict[particle])
        self.update_delStartEndLabel(particle)
        
    def update_delStartEndLabel(self,particle,*args):
        #Update label for Start/End frames
        startend = self.correct_startend[particle] 
        self.correct_delStart.set(startend[0]); self.correct_delEnd.set(startend[1])

    def update_delParticleList(self, *args):
        #Changes selectable particles
        particle_list = sorted(self.check_particle_list())
        self.entryDeleteParticle['values'] = particle_list
        self.entryJoinParticle1['values'] = particle_list
        
    def update_joinParticle2(self, *args):
        #Updates the particle to be joined based on previous modifications of the dictionary
        self.correct_joinParticle2.set(self.correct_joinDict[self.correct_joinParticle1.get()])        
        
    def update_joinDict(self, *args):
        #When Particle2 is changed, update joinDict with particle 2 being linked to particle 1
        p2 = self.correct_joinParticle2.get()
        if p2 != None:
            self.correct_joinDict[self.correct_joinParticle1.get()] = p2
        
    def update_joinParticle2List(self, *args):
        #Changes selectable particles to be joined
        p = self.correct_joinParticle1.get()
        p_end = self.correct_startend[p][1]
        p_list = self.check_particle_list()
        p2_list = []
        for p2 in p_list:
            if p != p2 and self.correct_startend[p2][0] > p_end:
                p2_list.append(str(p2))
            
        self.entryJoinParticle2['values'] = ['None'] + sorted(p2_list)
   
    def check_particle_list(self,*args):
        #Returns list of available particles based on infoDict
        part_list = self.get_particle_list()
        if not self.data_checkAll.get(): #Return available selected particles
            part_selected_list = dfun.interpret_list(self.data_ParticleList.get())
            return [p for p in part_selected_list if p in part_list]
        else: #Return all available particles
            return part_list
            
    def get_particle_list(self,*args):
        return [part for (part, info) in list(self.infoDict.keys()) if info == 'position'] #Returns available particles

    def get_startend_dic(self, infoDict, particle_list):
        startend_dic = {}
        for p in particle_list:
            startend = dfun.get_startend(infoDict[p,'position'])
            startend[1] = startend[1]*self.Vars['vid_skip']
            startend = [i + self.Vars['vid_start'] for i in startend]
            startend_dic[p] = startend
        return startend_dic
            
    def exit_data_corrector(self, *args):
        #Exit tasks related to corrector variables
        for p in list(self.correct_deleteDict.keys()):
            #Delete frames dict
            l = self.correct_deleteDict[p]
            isPselected = (self.data_checkAll.get()) or (p in dfun.interpret_list(self.data_ParticleList.get()))
            if not l or not isPselected:
                del self.correct_deleteDict[p] #Clear empty entries and non-selected particles
            else:
                self.correct_deleteDict[p] = dfun.interpret_list(l) #Interpret non-empty entries
                
            #Join trajectories dict
            if self.correct_joinDict[p] == 'None':
                del self.correct_joinDict[p]
            else:
                self.correct_joinDict[p] = int(self.correct_joinDict[p])
        


class chooseMethod(mainGUI):

    def __init__(self, root):

        #Define variables
        self.method = tkinter.StringVar() #Detection method (thresh,gradient,absThresh,bgextr)
        self.trackMethod = tkinter.StringVar() #Tracking method (auto,live,manual)
        self.angleInfo = tkinter.StringVar() #Angle information (nothing, jet, Janus, jet+bubble)
        self.angleZ = tkinter.BooleanVar() #Z angle (True or False)
        # for var in self.__dict__:
        #     print("%s : %s" % (var, self.__dict__[var]))
        self.variables = { var: self.__dict__[var] for var in self.__dict__} #Store only the Tkinter variables

        self.default_values = {'method': 'gradient', 'trackMethod':'auto', 'angleInfo':'nothing', 'angleZ':False}

        self.open_config('config_choose.cfg')

        #Set button keys
        root.bind('<Return>', self.exit_task) #Binds enter to the start button
        root.bind('<Escape>', lambda *args:sys.exit()) #Binds enter to the start button
        self.variables['trackMethod'].trace('w',self.check_trackMethod)

        self.root = root

        #Widgets
        self.mainframe = tkinter.ttk.Frame(root, padding = (10,10,10,10))
        self.buttonContinue = tkinter.ttk.Button(self.mainframe, text = 'Continue', command = self.exit_task)
        self.buttonContinue.state(['active'])
        self.buttonCancel = tkinter.ttk.Button(self.mainframe, text = 'Cancel', command = lambda :sys.exit())

        self.labelTrackChoose = tkinter.ttk.Label(self.mainframe, text = 'Choose tracking method: ')
        self.radioTrackAuto = tkinter.Radiobutton(self.mainframe, text = 'Auto tracking', variable = self.trackMethod, value = 'auto')
        self.radioTrackLive = tkinter.Radiobutton(self.mainframe, text = 'Live tracking', variable = self.trackMethod, value = 'live')
        self.radioTrackManual = tkinter.Radiobutton(self.mainframe, text = 'Manual tracking', variable = self.trackMethod, value = 'manual')

        self.labelChoose = tkinter.ttk.Label(self.mainframe, text = 'Choose detection method: ')
        self.buttonGradient = tkinter.Radiobutton(self.mainframe, text = 'Gradient', variable = self.method, value = 'gradient', indicatoron = 1)
        self.buttonThresh = tkinter.Radiobutton(self.mainframe, text = 'Threshold', variable = self.method, value = 'thresh', indicatoron = 1)
        self.buttonAbsThresh = tkinter.Radiobutton(self.mainframe, text = 'Abs. thresh', variable = self.method, value = 'absThresh', indicatoron = 1)
        self.buttonBgExtraction = tkinter.Radiobutton(self.mainframe, text = 'Bg extract', variable = self.method, value = 'bgextr', indicatoron = 1)

        self.labelExtraInfo = tkinter.ttk.Label(self.mainframe, text = 'Extra information: ')
        self.buttonNothing = tkinter.Radiobutton(self.mainframe, text = 'Nothing', variable = self.angleInfo, value = 'nothing', indicatoron = 1)
        self.buttonAngleJet = tkinter.Radiobutton(self.mainframe, text = 'Angle (jet)', variable = self.angleInfo, value = 'angleJet', indicatoron = 1)
        self.buttonAngleJanus = tkinter.Radiobutton(self.mainframe, text = 'Angle (Janus)', variable = self.angleInfo, value = 'angleJanus', indicatoron = 1)
        self.buttonAngleJetBubble = tkinter.Radiobutton(self.mainframe, text = 'Angle (jet + bubbles)', variable = self.angleInfo, value = 'angleJetBubble', indicatoron = 1)
        self.checkAngleZ = tkinter.ttk.Checkbutton(self.mainframe, text = 'Z angle', variable = self.angleZ, onvalue = True, offvalue = False)

        #Grid
        self.N, self.S, self.E, self.W = tkinter.N, tkinter.S, tkinter.E, tkinter.W
        self.mainframe.grid(column = 0, row = 0)
        self.labelTrackChoose.grid(column = 0, row = 0, columnspan = 2, pady = 5, padx = 5, sticky = self.W)
        self.radioTrackAuto.grid(column = 0, row = 1, sticky = (self.W), pady = 5, padx = 5)
        self.radioTrackLive.grid(column = 1, row = 1, sticky = (self.W), pady = 5, padx = 5)
        self.radioTrackManual.grid(column = 2, row = 1, sticky = (self.W), pady = 5, padx = 5)

        self.labelChoose.grid(column = 0, row = 2, columnspan = 2, pady = 5, padx = 5, sticky = self.W)
        self.buttonGradient.grid(column = 0, row = 3, sticky = (self.W, self.N, self.S), pady = (0,5), padx = 5)
        self.buttonThresh.grid(column = 1, row = 3, sticky = (self.W, self.N, self.S), pady = (0,5), padx = 5)
        self.buttonAbsThresh.grid(column = 2, row = 3, sticky = (self.W,self.N, self.S ), pady = (0,5), padx = 5)
        self.buttonBgExtraction.grid(column = 3, row = 3, sticky = (self.W, self.N, self.S), pady = (0,5), padx = 5)

        self.labelExtraInfo.grid(column = 0, row = 4, columnspan = 3, pady = 5, padx = 5, sticky = self.W)
        self.buttonNothing.grid(column = 0, row = 5, pady = (0,5), padx = 5, sticky = self.W)
        self.buttonAngleJet.grid(column = 1, row = 5, pady = (0,5), padx = 5, sticky = self.W)
        self.buttonAngleJanus.grid(column = 2, row = 5, pady = (0,5), padx = 5, sticky = self.W)
        self.buttonAngleJetBubble.grid(column = 3, row = 5, pady = (0,5), padx = 5, sticky = self.W)
        self.checkAngleZ.grid(column = 0, row = 6, pady = (0,5), padx = (7,0), sticky = self.W)

        self.buttonContinue.grid(column = 3, row = 7, sticky = self.E, pady = (15,5), padx = 5)
        self.buttonCancel.grid(column = 0, row = 7, sticky = self.W, pady = 5, padx = 5)
        #Main window
        root.title('Method selection')
        root.geometry('+100+100')

        self.check_trackMethod()

    def check_trackMethod(self, *args):
        #Checks selected track method and deactivates radiobuttons if necessary
        trackMethod = self.trackMethod.get()
        self.check_widget(trackMethod == 'manual', self.buttonGradient)
        self.check_widget(trackMethod == 'manual', self.buttonThresh)
        self.check_widget(trackMethod == 'manual', self.buttonAbsThresh)
        self.check_widget(trackMethod == 'manual', self.buttonBgExtraction)
        self.check_widget(trackMethod == 'manual', self.buttonNothing)
        self.check_widget(trackMethod == 'manual', self.buttonAngleJet)
        self.check_widget(trackMethod == 'manual', self.buttonAngleJanus)
        self.check_widget(trackMethod == 'manual', self.buttonAngleJetBubble)
        self.check_widget(trackMethod == 'manual', self.checkAngleZ)



########### SPECIFIC GUI CLASSES #############

 
class gradientGUI(mainGUI):

    def __init__(self, root, *args):
        mainGUI.__init__(self, root, *args) #Calls __init__ method from mainGUI

        self.sobel_order = tkinter.IntVar() #Order of sobel derivative
        self.sobel_ksize = tkinter.IntVar() #Size of kernel in sobel derivative (1,3,5,7 or -1)
        self.sobel_thresh = tkinter.IntVar() #Threshold applied to the sobel_derivative
        self.mode = 'gradient'

        self.variables.update({'sobel_order' : self.sobel_order, 'sobel_ksize' : self.sobel_ksize, 'sobel_thresh':self.sobel_thresh})

        self.default_values.update({'sobel_order' : 1, 'sobel_ksize' : 5, 'sobel_thresh':175})

        self.open_config('config_gradient.cfg')

       ##### Sobel options #####
        self.sobelframe = tkinter.ttk.Labelframe(self.mainframe, text = ' Gradient options: ', padding = 10, relief = 'sunken')
        #Entries
        self.entryOrder = tkinter.Spinbox(self.sobelframe, from_=1, to=5, textvariable = self.sobel_order, width = 2)
        self.entryKsize = tkinter.ttk.Combobox(self.sobelframe, textvariable = self.sobel_ksize, width = 2)
        self.entryKsize['values'] = (-1,1,3,5,7)
        self.entryThresh = tkinter.ttk.Entry(self.sobelframe, textvariable = self.sobel_thresh, width = 4)
        #Labels
        self.labelOrder = tkinter.ttk.Label(self.sobelframe, text = 'Derivative Order')
        self.labelKsize = tkinter.ttk.Label(self.sobelframe, text = 'Kernel size')
        self.labelThresh = tkinter.ttk.Label(self.sobelframe, text = 'Threshold')

        #Sobel options (entries)
        self.sobelframe.grid(column=0, row = 5, columnspan = 3, sticky = (self.N,self.S,self.E,self.W), pady = (10,0))
        self.entryOrder.grid(column = 3, row = 0, padx = (3,50))
        self.entryKsize.grid(column = 5, row = 0, padx = (3,10))
        self.entryThresh.grid(column = 1, row = 0, padx = (3,50))

        #Sobel options (labels)
        self.labelOrder.grid(column = 2, row = 0, sticky = self.W)
        self.labelKsize.grid(column = 4, row = 0, sticky = self.W)
        self.labelThresh.grid(column = 0, row = 0, sticky = self.W, padx = (10,0))

        self.windowTitle = 'tracking - Gradient'
        root.title(self.trackMethodTitle[args[1]] + self.windowTitle)




class threshGUI(mainGUI):
     def __init__(self, root, *args):
        mainGUI.__init__(self,root,*args) #Calls __init__ method from mainGUI

        self.thr_BonW = tkinter.BooleanVar() #Are the particles black on white background?
        self.thr_size = tkinter.IntVar() #Size of adaptive thresholding block
        self.thr_C = tkinter.IntVar() #Adaptive thresholding constant that will be subtracted
        self.thr_adType = tkinter.StringVar() #Type of adaptive thresholding (mean or gaussian)

        self.mode = 'thresh'

        self.variables.update({'thr_BonW':self.thr_BonW, 'thr_size':self.thr_size, 'thr_C':self.thr_C, 'thr_adType':self.thr_adType})

        self.default_values.update({'thr_BonW': True, 'thr_size': 11, 'thr_C':4, 'thr_adType':'Gaussian average'})

        self.open_config('config_thresh.cfg')

        ##### Thresholding options #####
        self.threshframe = tkinter.ttk.Labelframe(self.mainframe, text = ' Thresholding options: ', padding = 10, relief = 'sunken')
        #Entries
        self.checkThrBonW = tkinter.ttk.Checkbutton(self.threshframe, text = 'Black nanomotors', variable = self.thr_BonW, onvalue = True, offvalue = False)
        self.entryThrSize = tkinter.Spinbox(self.threshframe, from_=3, to_=99, increment_=2,textvariable = self.thr_size, width = 2)
        self.entryThrC = tkinter.Spinbox(self.threshframe, from_=1, to_=31, increment_ = 1, textvariable = self.thr_C, width = 2)
        self.entryThrAdType = tkinter.ttk.Combobox(self.threshframe, textvariable = self.thr_adType, width = 15)
        self.entryThrAdType['values'] = ('Gaussian average', 'Normal mean')
        self.entryThrAdType.state(['readonly'])

        #Labels
        self.labelThrSize = tkinter.ttk.Label(self.threshframe, text = 'Block size')
        self.labelThrC =  tkinter.ttk.Label(self.threshframe, text = 'Subtracting constant')
        # self.labelThrAdType =  ttk.Label(self.threshframe, text = '')

        #Thresholding options
        self.threshframe.grid(column=0, row = 5, columnspan = 3, sticky = (self.N,self.S,self.E,self.W), pady = (10,0))

        self.labelThrSize.grid(column = 0, row = 0, sticky = self.W, pady = 5, padx = (10,5))
        self.entryThrSize.grid(column = 1, row = 0, sticky = self.W, pady = 5, padx = (3,10))
        self.labelThrC.grid(column = 0, row = 1, sticky = self.W, pady = 5, padx = (10,5))
        self.entryThrC.grid(column = 1, row = 1, sticky = self.W, pady = 5, padx = (3,10))

        self.entryThrAdType.grid(column = 2, row = 0, columnspan = 2, sticky = (self.W), padx = (10,0), pady = 5)
        self.checkThrBonW.grid(column = 2, row = 1, sticky = self.W, pady = 5, padx = (10,2))

        #self.threshframe.columnconfigure(0, weight = 1)
        #self.threshframe.columnconfigure(1, weight = 1)
        self.threshframe.columnconfigure(2, weight = 2)

        self.windowTitle = 'tracking - Adaptive Threshold'
        root.title(self.trackMethodTitle[args[1]] + self.windowTitle)


class absThreshGUI(mainGUI):

     def __init__(self, root, *args):
        mainGUI.__init__(self,root, *args) #Calls __init__ method from mainGUI

        self.thr_BonW = tkinter.BooleanVar() #Are the particles black on white background?
        self.thr_thresh = tkinter.IntVar() #Size of adaptive thresholding block
        self.mode = 'absThresh'

        self.variables.update({'thr_BonW':self.thr_BonW, 'thr_thresh':self.thr_thresh})

        self.default_values.update({'thr_BonW': True, 'thr_thresh': 125})

        self.open_config('config_absThresh.cfg')

        ##### Thresholding options #####
        self.threshframe = tkinter.ttk.Labelframe(self.mainframe, text = ' Thresholding options: ', padding = 10, relief = 'sunken')
        #Entries
        self.checkThrBonW = tkinter.ttk.Checkbutton(self.threshframe, text = 'Black nanomotors', variable = self.thr_BonW, onvalue = True, offvalue = False)
        self.entryThresh = tkinter.Spinbox(self.threshframe, from_=0, to_=255, increment_=1,textvariable = self.thr_thresh, width = 3)

        #Labels
        self.labelThresh = tkinter.ttk.Label(self.threshframe, text = 'Threshold')

        #Thresholding options
        self.threshframe.grid(column=0, row = 5, columnspan = 3,sticky = (self.N,self.S,self.E,self.W), pady = (10,0))
        self.checkThrBonW.grid(column = 0, row = 0, columnspan = 2, sticky = (self.E,self.W), pady = 5, padx = (10,2))
        self.entryThresh.grid(column = 2, row = 0, sticky = self.E, pady = 5, padx = (10,2))
        self.labelThresh.grid(column = 3, row = 0, sticky = self.W, pady = 5, padx = (2,10))

        # self.threshframe.columnconfigure(0, weight = 2)
        # self.threshframe.columnconfigure(1, weight = 1)
        # self.threshframe.columnconfigure(2, weight = 2)
        # self.threshframe.columnconfigure(3, weight = 1)

        self.windowTitle = 'tracking - Basic Threshold'
        root.title(self.trackMethodTitle[args[1]] + self.windowTitle)


class bgextrGUI(mainGUI):

    #Specific GUI for background extraction

    def __init__(self, root, *args):
        mainGUI.__init__(self,root, *args) #Calls __init__ method from mainGUI

        self.bg_thresh = tkinter.IntVar() #Threshold applied on the subtracted image
        self.bg_actualize = tkinter.IntVar() #Time for the reference frame to actualize
        self.bg_bgColor = tkinter.IntVar() #Approximate grayscale value of background
        self.bg_refFrame = tkinter.IntVar() #Frame used as reference
        self.bg_mode = tkinter.StringVar() #Mode of background extraction: single frame, total average, dynamic average
        self.bg_alpha = tkinter.DoubleVar() #Alpha parameter in the dynamic average mode (ref = alpha * ref + (1-alpha) * frame)
        self.mode = 'bgextr'

        self.variables.update({'bg_thresh':self.bg_thresh, 'bg_actualize':self.bg_actualize, 'bg_bgColor':self.bg_bgColor, 'bg_refFrame':self.bg_refFrame, 'bg_mode':self.bg_mode, 'bg_alpha':self.bg_alpha})
        self.default_values.update({'bg_thresh':30, 'bg_open': 1, 'bg_close': 3, 'bg_actualize': 50, 'bg_bgColor':100, 'bg_refFrame':1, 'bg_mode':'Single Frame', 'bg_alpha':0.9})

        self.open_config('config_bgextr.cfg')
        self.bg_mode_id = {0:'Single Frame', 1:'Total Average', 2:'Dynamic Average'} #Dict relating tab index with name
        self.bg_mode_id.update({name:ind for ind, name in list(self.bg_mode_id.items())}) #Reversible dict
        self.bg_refFrame.trace('w',self.bg_reference_change)
        self.bg_reference = ['','',''] #Stores the references for all three modes
        self.bg_refOK = {'Single Frame':False, 'Total Average':False, 'Dynamic Average':False} # True if the correct reference frame is stored
        self.bg_lastRef = 0 # Frame number of the last time reference frame was actualized
        
        self.load_ref() #Load reference image
        
        #### BACKGROUND EXTRACTION OPTIONS ####
        self.bgframe = tkinter.ttk.Labelframe(self.mainframe, text = ' Background extraction options: ', padding = 10, relief = 'sunken')

        self.tabBG = tkinter.ttk.Notebook(self.bgframe, padding = 0)
        self.singleframemode = tkinter.ttk.Frame(self.tabBG)
        self.totalavmode = tkinter.ttk.Frame(self.tabBG)
        self.dynamicavmode = tkinter.ttk.Frame(self.tabBG)

        self.tabBG.add(self.singleframemode, text = 'Single Frame')
        self.tabBG.add(self.totalavmode, text = 'Total Average')
        self.tabBG.add(self.dynamicavmode, text = 'Dynamic Average')

        #Single frame reference
        self.labelThresh = tkinter.ttk.Label(self.singleframemode, text = 'Threshold')
        self.entryThresh = tkinter.Spinbox(self.singleframemode, from_ = 0, to_ = 255, increment_ = 1, textvariable = self.bg_thresh, width = 3)
        self.labelBgActualize = tkinter.ttk.Label(self.singleframemode, text = 'Reference update (frames)')
        self.entryBgActualize = tkinter.Spinbox(self.singleframemode, from_=1, to_=10000,increment_=1,textvariable=self.bg_actualize, width=3)
        self.labelBgColor = tkinter.ttk.Label(self.singleframemode, text = 'Background color')
        self.entryBgColor = tkinter.Spinbox(self.singleframemode, from_ = 0, to_ = 255, increment_=1, textvariable = self.bg_bgColor, width = 4)
        self.labelRefFrame = tkinter.ttk.Label(self.singleframemode, text = 'Reference frame')
        self.entryRefFrame = tkinter.ttk.Entry(self.singleframemode, textvariable = self.bg_refFrame, width = 4)
        
        #Total average reference
        self.labelThreshAvRef = tkinter.ttk.Label(self.totalavmode, text = 'Threshold')
        self.entryThreshAvRef = tkinter.Spinbox(self.totalavmode, from_ = 0, to_ = 255, increment_ = 1, textvariable = self.bg_thresh, width = 3)
        
        #Dynamic average reference
        self.labelThreshDynRef = tkinter.ttk.Label(self.dynamicavmode, text = 'Threshold')
        self.entryThreshDynRef = tkinter.Spinbox(self.dynamicavmode, from_ = 0, to_ = 255, increment_ = 1, textvariable = self.bg_thresh, width = 3)
        self.labelAlpha = tkinter.ttk.Label(self.dynamicavmode, text = 'Alpha')
        self.entryAlpha = tkinter.Spinbox(self.dynamicavmode, from_ = 0, to_ = 1, increment_ = 0.01, textvariable = self.bg_alpha, width = 4)
        
        
        #### Grid ####
        colsep = 10
        #Single frame reference
        self.bgframe.grid(column=0, row = 5, columnspan = 4, rowspan = 2, sticky = (self.N,self.S,self.E,self.W))
        self.tabBG.grid(column=0, row = 0, sticky = (self.N,self.S,self.E,self.W))
        self.labelThresh.grid(column = 0, row = 0, sticky = self.W, pady = (10,5), padx = (2,5))
        self.entryThresh.grid(column = 1, row = 0, sticky = self.E, pady = (10,5), padx = (5,colsep))

        self.labelRefFrame.grid(column = 2, row = 0, sticky = (self.N,self.S,self.W), pady = 5, padx = (2,5))
        self.entryRefFrame.grid(column = 3, row = 0, sticky = self.W, pady = 5, padx = (5,colsep))
        self.labelBgActualize.grid(column = 0, row = 1, sticky = self.W, pady = 5, padx = (2,5))
        self.entryBgActualize.grid(column = 1, row = 1, sticky = self.W, pady = 5, padx = (5,colsep))
        self.labelBgColor.grid(column = 2, row = 1, sticky = self.W, pady = 5, padx = (2,5))
        self.entryBgColor.grid(column = 3, row = 1, sticky = self.W, pady = 5, padx = (5,50))              

        #Total average reference
        self.labelThreshAvRef.grid(column = 0, row = 0, sticky = self.W, pady = (10,5), padx = (2,5))
        self.entryThreshAvRef.grid(column = 1, row = 0, sticky = self.E, pady = (10,5), padx = (5,colsep))

        #Dynamic average reference
        self.labelThreshDynRef.grid(column = 0, row = 0, sticky = self.W, pady = (10,5), padx = (2,5))
        self.entryThreshDynRef.grid(column = 1, row = 0, sticky = self.E, pady = (10,5), padx = (5,colsep))
        self.labelAlpha.grid(column = 2, row = 0, sticky = self.W, pady = (10,5), padx = (colsep, 5))
        self.entryAlpha.grid(column = 3, row = 0, sticky = self.E, pady = (10,5), padx = (colsep, 5))
        
        if self.trackMethod == 'live':
            #Single Frame can't be used in live tracking Method
            self.tabBG.tab(0, state = 'disabled')


        self.tabBG.select(self.bg_mode_id[self.bg_mode.get()])
        self.tabBG.bind_all("<<NotebookTabChanged>>", self.change_tab)

        self.filename.trace('w',self.change_filename)

        self.windowTitle = 'tracking - Background extraction'
        root.title(self.trackMethodTitle[args[1]] + self.windowTitle)
        
    def change_filename(self, *args):
        #If the filename is changed, then the ref image is not useful anymore
        self.bg_refOK['Single Frame'] = self.bg_refOK['Total Average'] = self.bg_refOK['Dynamic Average'] = False
    
    
    def change_tab(self,*args):
        #Updates the bg_mode variable when the tab changes
        new_mode = self.bg_mode_id[self.tabBG.index('current')]
        self.bg_mode.set(new_mode)
        if new_mode == 'Total Average':
            self.previewButton.state(['!disabled'])
        else:
            self.previewButton.state(['disabled'])
            if self.isPreviewOn:
                self.close_preview()
        

    def get_reference(self,*args):
        
        vid = cv2.VideoCapture(self.variables['filename'].get())   
        if self.bg_mode.get() == 'Single Frame':     
            
            #Runs the whole video and stores the selected reference frame
            if self.isCv2Accurate == False:
                #Search for the frame manually
                count = 0
                while True:
                    if count in [1]:
                        #Opening a window after vid.read() makes a "segmentation fault" error not appear with weird codecs
                        cv2.namedWindow('temp')
                        cv2.imshow('temp', np.ones((1,1,1)))
                        cv2.waitKey(1)
                        cv2.destroyAllWindows()
                        cv2.imshow('temp', np.ones((1,1,1)))
                    ret, frame = vid.read()
                    if ret == False:
                        break
                    count += 1
                    if count == self.bg_refFrame.get():
                        #Store reference background
                        break
    
            else:
                self.vid.set(1, self.bg_refFrame.get()-1)
                ret, frame = self.vid.read()
            
            reference = frame
            self.bg_set_reference(reference, 'Single Frame', count = count)
                
        elif self.bg_mode.get() == 'Total Average':
            # Creates a reference image by averaging the whole video
            count = 0
            width,height = int(vid.get(3)),int(vid.get(4))
            reference = np.zeros((height, width, 3), dtype = int)
            while True:
                ret, frame = vid.read()
                if ret == False:
                    break
                
                count += 1
                reference += frame
            # [modified] error in py3 (or numpy 1.12), cause type error
            # reference /= float(count)
            reference = reference/float(count)
            reference = reference.astype(np.uint8)
            self.bg_set_reference(reference, 'Total Average', count = count)

        
        elif self.bg_mode.get() == 'Dynamic Average':
            #For the dynamic average, we start with the first frame
            ret, frame = vid.read()
            reference = frame
            self.bg_set_reference(reference, 'Dynamic Average')
            
        return reference

    def bg_set_reference(self, reference, mode, count = 0):
        #Count is only given for the Single Frame case
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.bg_reference[self.bg_mode_id[mode]] = reference
        self.bg_refOK[mode] = True
        if mode == 'Total Average':
            self.vid_totalFrames.set(count) #Save total number of frames
        elif mode == 'Single Average':
            self.lastRef = count

    def bg_reference_change(self,*args):
        #Keep track of changes in the reference frame number

        try:        
            if self.bg_refFrame.get() != self.bg_lastRef:
                self.bg_refOK['Single Frame'] = False
            else:
                self.bg_refOK['Single Frame'] = True

        except ValueError:
            #When the value of self.bg_refFrame is left blank
            return

    def exit_task(self,*args):
        #Search for the reference frame
        mode = self.bg_mode.get()
        if not self.bg_refOK[mode]:
            self.get_reference()
            
        if self.bg_refOK['Total Average']:
            np.savez('bgextr_ref', self.filename.get(), self.bg_reference[1]) #Save reference image of Total Average
            
        self.bg_reference = self.bg_reference[self.bg_mode_id[mode]] #Only retains the reference of the selected mode
        mainGUI.exit_task(self)
        #super(bgextrGUI, self).exit_task()

    def load_ref(self, *args):
        #Load reference image. 'arr_0' is video filename and 'arr_1' is ref image
        name = 'bgextr_ref.npz'
        if os.path.isfile(name):
            data = np.load(name)
            filename = data['arr_0']
            if filename == self.filename.get():
                #If it's the same video, get reference image
                self.bg_reference[1] = data['arr_1']
                self.bg_refOK['Total Average'] = True


class manualTrackGUI(mainGUI): #, *args, **kwargs):

    def __init__(self, root, *args, **kwargs):

        self.__init_basic__() #Creates default dict
        self.manual_store = tkinter.IntVar() #Amount of frames that can be stored at maximum
        self.manual_show_lines = tkinter.BooleanVar() #Show previous lines on screen (can obstruct view)
        self.manual_angle = tkinter.BooleanVar() #Manually indicate the angle

        # Dictionary of relevant variables accessed through their name
        self.variables = self.create_tkinter_dict()

        self.default_values.update({
        'manual_store': 500,
        'manual_show_lines':True,
        'manual_angle':False})

        self.open_config('config_manual.cfg')
        
        self.root = root
        root.bind('<Return>', self.exit_task) #Binds enter to the start button
        root.bind('<Escape>', lambda *args:sys.exit()) #Binds ESC to the exit button

        self.N, self.S, self.E, self.W = tkinter.N, tkinter.S, tkinter.E, tkinter.W

        ##### Main frame  #####
        self.mainframe = tkinter.ttk.Frame(root, padding = (10,10,10,5))
        self.videolbl = tkinter.ttk.Label(self.mainframe, text = 'Choose video:')
        self.entryVideo = tkinter.ttk.Entry(self.mainframe, textvariable = self.filename, width = 40)
        self.videoButton = tkinter.ttk.Button(self.mainframe, text = 'Search...', command = self.get_filename)
        self.acceptButton = tkinter.ttk.Button(self.mainframe, text = 'Start', command = self.exit_task )
        self.acceptButton.state(['active'])
        root.bind('<Return>', self.exit_task) #Binds enter to the start button
        self.defaultButton = tkinter.ttk.Button(self.mainframe, text = 'Set default values  ', command = self.set_default)
        self.cancelButton = tkinter.ttk.Button(self.mainframe, text = 'Cancel', command = lambda :sys.exit())
        self.previousButton = tkinter.ttk.Button(self.mainframe, text = 'Set previous values', command = lambda : self.read_config('Previous_values'))
        self.goBackButton = tkinter.ttk.Button(self.mainframe, text = 'Go back', width = 10, command  = self.go_back)

        self.valuebuttonframe = tkinter.ttk.Frame(self.mainframe, padding = (1,1,1,1), relief = 'sunken')
        self.defaultButton = tkinter.ttk.Button(self.valuebuttonframe, text = 'Set default', command = self.set_default)
        self.previousButton = tkinter.ttk.Button(self.valuebuttonframe, text = 'Set previous', command = self.read_previous_config)
        
        #Scrollable entry
        self.entryVideo.focus()
        self.entryVideoScroll = tkinter.ttk.Scrollbar(self.mainframe, orient = tkinter.HORIZONTAL, command = self.scrollHandler)
        self.entryVideo['xscrollcommand'] = self.entryVideoScroll.set
        self.entryVideo.xview_moveto(1)

        ##### Video options #####
        self.videoframe = tkinter.ttk.Labelframe(self.mainframe, text= ' Video options: ', padding = 10, relief = 'sunken')
        self.checkFit = tkinter.ttk.Checkbutton(self.videoframe, text = 'Fit to screen', variable = self.vid_fit, onvalue = True, offvalue = False)
        self.checkLines = tkinter.ttk.Checkbutton(self.videoframe, text = 'Show lines', variable = self.manual_show_lines, onvalue = True, offvalue = False)
        self.labelSkip = tkinter.ttk.Label(self.videoframe, text = 'Skip frames ')
        self.entrySkip = tkinter.ttk.Entry(self.videoframe, textvariable = self.vid_skip, width = 4)
        self.labelVidStart = tkinter.ttk.Label(self.videoframe, text = 'Start frame ')
        self.entryVidStart = tkinter.ttk.Entry(self.videoframe, textvariable = self.vid_start, width = 4)
        self.labelVidEnd = tkinter.ttk.Label(self.videoframe, text = 'End frame ')
        self.entryVidEnd = tkinter.ttk.Entry(self.videoframe, textvariable = self.vid_end, width = 4)
        self.labelFPS = tkinter.ttk.Label(self.videoframe, text = 'Video FPS')
        self.labelFPS = tkinter.ttk.Button(self.videoframe, text = 'Video FPS', command = lambda :self.calculate_FPS_window('vid'))
        self.entryFPS = tkinter.ttk.Entry(self.videoframe, textvariable = self.vid_FPS, width = 4)


        ##### Manual track options #####
        self.manualframe = tkinter.ttk.Labelframe(self.mainframe, text = ' Manual tracking options: ', padding = 10, relief = 'sunken')
        self.labelStore = tkinter.ttk.Label(self.manualframe, text = 'Number of stored frames: ')
        self.entryStore = tkinter.Spinbox(self.manualframe, from_=0, to= 100000, textvariable = self.manual_store, width = 5)
        self.checkAngle = tkinter.ttk.Checkbutton(self.manualframe, text = 'Indicate orientation', variable = self.manual_angle, onvalue = True, offvalue = False)

        ##### Grid configuration #####
        lastrow = 8
        self.mainframe.grid(column=0, row=0, sticky = (self.N,self.S,self.E,self.W))
        self.videolbl.grid(column = 0, row = 0, sticky = self.W)
        self.entryVideo.grid(column = 0, row = 1, padx = (3,10), sticky = (self.E,self.W), columnspan = 2 )
        self.videoButton.grid(column = 2, row = 1, sticky = self.E)
        self.entryVideoScroll.grid(column = 0, row = 2, pady = (0,15), padx = (3,10), sticky = (self.E,self.W), columnspan = 2)

        self.acceptButton.grid(column = 2, row = lastrow+1, pady = (0,5), sticky = (self.E, self.W))
        self.cancelButton.grid(column = 1, row = lastrow+1, sticky = self.E, pady = (0,5), padx = (0, 15))
        self.previousButton.grid(column = 0, row = lastrow+1, pady = (0,5), padx = (0,10), sticky = (self.W))
        self.defaultButton.grid(column = 0, row = lastrow, pady = (10,0), padx = (0,15), sticky = (self.W))
        self.goBackButton.grid(column = 0, row = lastrow+1, pady = (5,5), padx = (10,0), sticky = (self.W))

        self.valuebuttonframe.grid(column = 0, row = 8, pady = (5,5), padx = (0,0), sticky = self.W)
        self.defaultButton.grid(column = 0, row = 0, pady = (0,0), padx = (0,0), sticky = (self.W))
        self.previousButton.grid(column = 1, row = 0, pady = (0,0), padx = (0,0), sticky = (self.W))

        #Video options
        self.videoframe.grid(column = 0, row = 2, rowspan = 2, columnspan = 6, sticky = (self.N,self.S,self.E,self.W), pady = (10,0))
        self.labelVidStart.grid(column = 0, row = 0, padx = (10,3), pady = (0,10), sticky = self.W)
        self.entryVidStart.grid(column = 1, row = 0, padx = (3,10), pady = (0,10), sticky = self.W)
        self.labelVidEnd.grid(column = 2, row = 0 , padx = (10,3), pady = (0,10), sticky = self.W)
        self.entryVidEnd.grid(column = 3, row = 0, padx = (3,10) , pady = (0,10), sticky = self.W)
        self.labelSkip.grid(column = 0, row = 1, padx = (10,3), pady = (0,10), sticky = self.W)
        self.entrySkip.grid(column = 1, row = 1, padx = (3,10), pady = (0,10), sticky = self.E)
        self.labelFPS.grid(column = 2, row = 1, padx = (10,3), pady = (0,10), sticky = self.W)
        self.entryFPS.grid(column = 3, row = 1, padx = (3,10), pady = (0,10), sticky = self.E)
        self.checkFit.grid(column = 4, row = 1, padx = 10, pady = (0,10), sticky = self.W)
        self.checkLines.grid(column = 4, row = 0, padx = 10, pady = (0,10), sticky = self.W)

        # self.videoframe.columnconfigure(0, weight = 2)
        # self.videoframe.columnconfigure(1, weight = 2)


        #Manual tracking options
        leftSpace = 10
        self.manualframe.grid(column = 0, row = 4, rowspan = 3, columnspan = 4, sticky = (self.W, self.E), pady = (10,10))
        self.checkAngle.grid(column = 0, row = 0, columnspan = 2, sticky = self.W, pady = 5, padx = (leftSpace,10))
        self.labelStore.grid(column = 0, row = 2, columnspan = 1, sticky = self.W, pady = 5, padx = (leftSpace,2))
        self.entryStore.grid(column = 1, row = 2, sticky = self.W, pady = 5, padx = (2,10))

        # self.manualframe.columnconfigure(0, weight = 1)
        # self.manualframe.columnconfigure(1, weight = 1)
        # self.manualframe.columnconfigure(2, weight = 1)


        self.file_opt = self.options = {}
        self.options['filetypes'] = [('Video files', '.avi'),
                                     ('Video files', '.wmv'),
                                     ('Video files', '.mp4'),
                                     ('All files', '.*')]

        #Main window
        self.windowTitle = ''
        root.title(self.windowTitle)

class EndScreen(mainGUI):
    #This window appears at the end of the script

    def __init__(self, root):

        self.root = root
        self.loopProcessing = True #Loop post-processing
        self.loopTrack = True #Loop to track menu

        root.bind('<Return>', self.exit_task) #Binds enter to the start button
        root.bind('<Escape>', lambda *args:sys.exit()) #Binds ESC to the exit button

        self.N, self.S, self.E, self.W = tkinter.N, tkinter.S, tkinter.E, tkinter.W

        self.mainframe = tkinter.ttk.Frame(self.root, padding = 10)
        self.exitLabel = tkinter.ttk.Label(self.mainframe, text = 'What do you want to do?')
        self.exitButton = tkinter.ttk.Button(self.mainframe, text = 'Exit', command = sys.exit )
        self.mainMenuButton = tkinter.ttk.Button(self.mainframe, text = 'Go to main menu', command = self.go_back_menu)
        self.trackMenuButton = tkinter.ttk.Button(self.mainframe, text = 'Go to track menu', command = self.go_back_track)
        self.postProcessingButton = tkinter.ttk.Button(self.mainframe, text = 'Go to post-processing', command = self.go_back_processing)


        self.mainframe.grid(column = 0, row = 0, columnspan = 2)
        self.exitLabel.grid(column = 0, row = 0, padx = 5, pady = 5)
        self.postProcessingButton.grid(column = 0, row = 1, padx = 5, pady = 5)
        self.trackMenuButton.grid(column = 1, row = 1, padx = 5, pady = 5)
        self.mainMenuButton.grid(column = 2, row = 1, padx = 5, pady = 5)
        self.exitButton.grid(column = 3, row = 1, padx = 5, pady = 5)

        root.geometry('+50+50')
        root.title('Exit screen')

    def go_back_processing(self):
        #Exit and continue post-processing loop
        self.root.destroy()

    def go_back_track(self):
        #Exit and continue track menu loop
        self.root.destroy()
        self.loopProcessing = False

    def go_back_menu(self):
        #Exit and continue main menu loop
        self.root.destroy()
        self.loopProcessing = False
        self.loopTrack = False






def TKimg_from_array(image):
    #Transforms numpy array image to a tk-compatible image
    if len(image.shape)>2:
        #BGR to RGB
        image[:,:,[0,1,2]] = image[:,:,[2,1,0]]
    PILimage = Image.fromarray(image)
    return ImageTk.PhotoImage(PILimage)

def start_program(trackMethod = None, mode = None, GUI = True):
    import detection_main
    detection_main.start_program(trackMethod = trackMethod, mode = mode, GUI = GUI)

if __name__ == '__main__':
    start_program()

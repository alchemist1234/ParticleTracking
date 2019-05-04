import tkinter
from tkinter.filedialog import *
import tkinter.ttk
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class chooseInterpolation(tkinter.Frame):
    #Choose interpolation method in case time intervals are variable
    def __init__(self, root):
        self.root = root

        self.method = tkinter.StringVar()
        self.n = tkinter.IntVar()
        self.show = tkinter.BooleanVar()

        self.method.set('')
        self.n.set(1000)
        self.show.set(True)

        #Widgets
        self.mainframe = tkinter.ttk.Frame(root, padding = (10,10,10,10))
        self.labelChoose = tkinter.ttk.Label(self.mainframe, text = 'Choose method of interpolation: ')
        self.buttonGradient = tkinter.ttk.Button(self.mainframe, text = 'Cubic spline', command = lambda method = 'cubic': self.exit_task(method))
        self.buttonThresh = tkinter.ttk.Button(self.mainframe, text = 'Linear', command = lambda method = 'linear': self.exit_task(method))

        self.optionsframe = tkinter.ttk.Frame(self.mainframe)
        self.labelN = tkinter.ttk.Label(self.optionsframe, text = 'Number of points')
        self.entryN = tkinter.ttk.Entry(self.optionsframe, textvariable = self.n, width = 5)
        self.checkShow = tkinter.ttk.Checkbutton(self.optionsframe, text = 'Show figure', variable = self.show, onvalue = True, offvalue = False)

        #Grid
        self.N, self.S, self.E, self.W = tkinter.N, tkinter.S, tkinter.E, tkinter.W
        self.mainframe.grid(column = 0, row = 0)
        self.labelChoose.grid(column = 0, row = 0, columnspan = 2, pady = 5, padx = 5)
        self.buttonGradient.grid(column = 0, row = 1, sticky = (self.E, self.W), pady = 5, padx = 5)
        self.buttonThresh.grid(column = 1, row = 1, sticky = (self.E, self.W), pady = 5, padx = 5)

        self.optionsframe.grid(column = 0, row = 2, columnspan = 2)
        self.labelN.grid(column = 0, row = 0, sticky = (self.E), pady = 5, padx = 5)
        self.entryN.grid(column = 1, row = 0, sticky = (self.W), pady = 5, padx = 5)
        self.checkShow.grid(column = 3, row = 0, pady = 5, padx = 5)

        #Main window
        root.title('Interpolation selection')
        root.geometry('+100+100')


    def exit_task(self, method):
        self.method.set(method)
        self.root.destroy()


class chooseMethod(tkinter.Frame):
    #Specify wether or not the time intervals are constant 
    def __init__(self, root):
        self.root = root

        self.method = tkinter.StringVar()
        self.method.set('')

        #Widgets
        self.mainframe = tkinter.ttk.Frame(root, padding = (10,10,10,10))
        self.labelChoose = tkinter.ttk.Label(self.mainframe, text = 'Your time intervals are...: ')
        self.buttonConstant = tkinter.ttk.Button(self.mainframe, text = 'Constant', command = lambda method = 'constant': self.exit_task(method))
        self.buttonVariable = tkinter.ttk.Button(self.mainframe, text = 'Variable', command = lambda method = 'variable': self.exit_task(method))

        #Grid
        self.N, self.S, self.E, self.W = tkinter.N, tkinter.S, tkinter.E, tkinter.W
        self.mainframe.grid(column = 0, row = 0)
        self.labelChoose.grid(column = 0, row = 0, columnspan = 2, pady = 5, padx = 5)
        self.buttonConstant.grid(column = 0, row = 1, sticky = (self.E, self.W), pady = 5, padx = 5)
        self.buttonVariable.grid(column = 1, row = 1, sticky = (self.E, self.W), pady = 5, padx = 5)

        #Main window
        root.title('Method selection')
        root.geometry('+100+100')


    def exit_task(self, method):
        self.method.set(method)
        self.root.destroy()


def mean_square_displacement(xvector,yvector):
    #Input: vector with 2d positions in tuples
    #Output: mean square displacement given by MSD(p) = sum( (r(i+p)-r(i))**2)/total
    length = len(xvector)
    intList = np.arange(1,length) #intervals
    MSD = np.arange(1,length, dtype = float) #To save the MSD values
    for interval in intList:
        intVector = [1]+[0]*(interval-1)+[-1] #Ex: for int = 3 you have [1,0,0,-1]
        #With "valid" you only take the overlapping points of the convolution
        convolutionx = np.convolve(xvector,intVector,'valid')
        convolutiony = np.convolve(yvector,intVector,'valid')
        MSDlist = convolutionx**2+convolutiony**2
        MSD[interval-1] = np.average(MSDlist)
    return intList,MSD

#Method GUI
root = tkinter.Tk()
chooseMethod = chooseMethod(root)
root.mainloop()

#Select file
options = {}
options['filetypes'] = [('CSV files', '.csv'),('All files', '.*')]
filename = askopenfilename(**options)
data = pd.read_csv(filename)
X, Y = list(data.X), list(data.Y)

if chooseMethod.method.get() == 'variable':
    #If the time intervals are variable we interpolate
    root = tkinter.Tk()
    chooseInterp = chooseInterpolation(root)
    root.mainloop()
    
    T = list(data.t)

    #Interpolation
    mode = chooseInterp.method.get()
    fInterX = interp1d(T, X, kind = mode)
    fInterY = interp1d(T, Y, kind = mode)
    N = chooseInterp.n.get()
    interT = np.linspace(T[0], T[-1], N)
    interX, interY = fInterX(interT), fInterY(interT)

    #MSD on the interpolated function
    intT, MSD = mean_square_displacement(interX,interY)

    #Normalization
    intT = intT * (T[-1]-T[0])/float(N)

    #Plot interpolation (if specified)
    if chooseInterp.show.get():
        plt.figure(1)
        plt.plot(T, X, 'o', label = 'X')
        plt.plot(T, Y, 'ro', label = 'Y')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Position (micrometers)')
        plt.legend()

        plt.figure(3)
        plt.plot(interT, interX, 'g', label = 'X interpolated')
        plt.plot(interT, interY, 'y', label = 'Y interpolated')
        interT = np.linspace(T[0], T[-1], 75)
        plt.plot(interT, fInterX(interT), 'o', label = 'X')
        plt.plot(interT, fInterY(interT), 'ro', label = 'Y')

        plt.xlabel('Time (seconds)')
        plt.ylabel('Position (micrometers)')
        plt.legend()

else:

    plt.plot(X, 'o', label = 'X')
    plt.plot(Y, 'ro', label = 'Y')

    # plt.plot(interT, interX, 'g', label = 'X interpolated')
    # plt.plot(interT, interY, 'y', label = 'Y interpolated')
    plt.xlabel('Time (frames)')
    plt.ylabel('Position (micrometers)')
    plt.legend()

    intT, MSD = mean_square_displacement(X,Y)


#Plot MSD
plt.figure(2)
plt.plot(intT, MSD,'o')
plt.xlabel('displacement')
plt.ylabel('MSD')
plt.show()

#Save new .csv file
parts = filename.split('/')
path = '/'.join(parts[:-1])
name = parts[-1][:-4]

with open(path+'/'+name+'_MSD.csv','wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['X','Y','Displacement','MSD'])
    posLen = len(X)
    posTime = len(intT)
    for i in range(posTime):
        if i < posLen:
            writer.writerow([X[i],Y[i],intT[i],MSD[i]])
        else:
            writer.writerow(['','',intT[i],MSD[i]])

import tkinter  # GUI interface
import tkinter.ttk  # GUI interface
import GUI_main  # GUI classes


def start_program(trackMethod=None, mode=None, GUI=True, extraInfo='nothing'):
    while (True):

        if GUI:
            # Starts the program using the selected options
            root = tkinter.Tk()
            # Start program menu
            chooseMethod = GUI_main.chooseMethod(root)
            if trackMethod != None:
                chooseMethod.trackMethod.set(trackMethod)

            root.mainloop()

            trackMethod = chooseMethod.trackMethod.get()  # Auto, live or manual
            mode = chooseMethod.method.get()  # Gradient, thresh, etc.
            extraInfo = [chooseMethod.angleInfo.get()]  # Nothing, jet, Janus, jet+bubble
            if 'nothing' in extraInfo:
                del extraInfo[extraInfo.index('nothing')]

            if chooseMethod.angleZ.get():
                extraInfo.append('angleZ')  # angleZ

        if trackMethod == 'auto':
            import detection_auto
            TrackProgram = detection_auto.class_factory(mode, extraInfo=extraInfo)  # Creates an auto class instance
            TrackProgram().run_main()  # Runs it

        elif trackMethod == 'live':
            import detection_live
            TrackProgramLive = detection_live.class_factory_live(mode,
                                                                 extraInfo=extraInfo)  # Creates a live class instance
            TrackProgramLive().run_main()  # Runs it

        elif trackMethod == 'manual':
            import detection_manual
            TrackProgramManual = detection_manual.class_factory_manual('manual', extraInfo=[
                'nothing'])  # Creates a manual class instance
            TrackProgramManual().run_main()  # Runs it

        GUI = True;
        trackMethod = None;
        mode = None  # Reset to main menu


if __name__ == '__main__':
    start_program()

"""
This program detects and reports objects from a picture.

@author: Emir Çetin MEMİŞ
@contact: memise@mef.edu.tr
@since: 2020-05-10

@TO-DO:
    - Implementation of the object detection algorithm.
    - Creating GUI for the application.
    - Designing database for the application.
    - Creating a picture creator for the application.
"""

from distutils.log import info
import cv2      as opencv
from cv2 import threshold
import numpy    as np
import tkinter  as tk
from   tkinter  import ANCHOR, ttk

class ObjectDetection(tk.Tk) :

    _objectNameFilePath      = "Assets/Object Class Names.txt"
    _configurationsFilePath  = "Assets/Configurations.pbtxt"
    _weightsFilePath         = "Assets/Weights.pb"

    def __init__(self, *args, **kwargs) :
        super().__init__(*args, **kwargs)

        self.objectTreshold = tk.DoubleVar(value=0.45)
        self.nmsThreshold   = tk.DoubleVar(value=0.2)
        self.captureDevice  = opencv.VideoCapture(0)
        self.allObjectNames = [cName[0:-1] for cName in open(self._objectNameFilePath,'r')]

        self.originalImage = None
        self.capturedImage = None

        self.loadCapModel()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.configure(background="black")

        self.parseContainer = tk.Frame(self)
        self.parseContainer.grid(row=0,column=0) 

        self.menubar = MenuBar(self, self.parseContainer)
        self.menubar.grid(row=0,column=0)

        self.output = Visualizer(self, self.parseContainer)
        self.output.grid(row=2,column=0)

        for children in self.parseContainer.winfo_children() :
            children.grid_configure(padx=5, pady=5)


    def loadCapModel(self) :

        self.capModel = opencv.dnn_DetectionModel(self._weightsFilePath,self._configurationsFilePath)
        self.capModel.setInputSize(320,320)
        self.capModel.setInputScale(1.0/ 127.5)
        self.capModel.setInputMean((127.5, 127.5, 127.5))
        self.capModel.setInputSwapRB(True)

    def proccesCapture(self) :

        failrue, self.capturedImage = self.captureDevice.read()

        self.originalImage = self.capturedImage.copy()

        objectIDs, confidences, bboxes = self.capModel.detect(self.capturedImage,confThreshold=self.objectTreshold.get())

        bboxes = list(bboxes)
        confidences = list(np.array(confidences).reshape(1,-1)[0])
        confidences = list(map(float,confidences))
        
        stats = opencv.dnn.NMSBoxes(bboxes,confidences,self.objectTreshold.get(),self.nmsThreshold.get())

        for i in stats:
            i = int(i)
            box = bboxes[i]
            x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
            opencv.rectangle(self.capturedImage, (x1,y1),(x1+x2,y2+y1), color=(0, 255, 0), thickness=2)
            opencv.putText(self.capturedImage,self.allObjectNames[objectIDs[i]-1].upper(),(box[0]+10,box[1]+30), opencv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

class MenuBar(ttk.Frame) :

    def __init__(self, root, parent, *args, **kwargs) :
        super().__init__(parent, *args, **kwargs)

        self.root = root

        self.firstObjectName    = tk.StringVar(value="")
        self.secondObjectName   = tk.StringVar(value="")
        self.thirdObjectName    = tk.StringVar(value="")
        self.fourthObjectName   = tk.StringVar(value="")
        self.fifthObjectName    = tk.StringVar(value="")

        """ OBJECT SELECTION """
        longestWorldLength = max(map(len,self.root.allObjectNames))

        firstObjectNameLabel = ttk.Label(self, text="1. OK Object:", foreground="light green", background="black")
        firsObjectNameCombobox = ttk.Combobox(self, textvariable=self.firstObjectName, values=self.root.allObjectNames, cursor="hand2", justify=tk.CENTER, width=longestWorldLength, )
        
        secondObjectNameLabel = ttk.Label(self, text="2. OK Object:", foreground="light green", background="black")
        secondObjectNameCombobox = ttk.Combobox(self, textvariable=self.secondObjectName, values=self.root.allObjectNames, cursor="hand2", justify=tk.CENTER, width=longestWorldLength)

        thirdObjectNameLabel = ttk.Label(self, text="3. OK Object:", foreground="light green", background="black")
        thirdObjectNameCombobox = ttk.Combobox(self, textvariable=self.thirdObjectName, values=self.root.allObjectNames, cursor="hand2", justify=tk.CENTER, width=longestWorldLength)

        fourthObjectNameLabel = ttk.Label(self, text="4. Not OK Object:", foreground="red", background="black")
        fourthObjectNameCombobox = ttk.Combobox(self, textvariable=self.fourthObjectName, values=self.root.allObjectNames, cursor="hand2", justify=tk.CENTER, width=longestWorldLength)

        fifthObjectNameLabel = ttk.Label(self, text="5. Not OK Object:", foreground="red", background="black")
        fifthObjectNameCombobox = ttk.Combobox(self, textvariable=self.fifthObjectName, values=self.root.allObjectNames, cursor="hand2", justify=tk.CENTER, width=longestWorldLength)

        comboboxses = [firsObjectNameCombobox, secondObjectNameCombobox, thirdObjectNameCombobox, fourthObjectNameCombobox, fifthObjectNameCombobox]

        firstObjectNameLabel.grid(row=0,column=0)
        firsObjectNameCombobox.grid(row=0,column=1)
        secondObjectNameLabel.grid(row=1,column=0)
        secondObjectNameCombobox.grid(row=1,column=1)
        thirdObjectNameLabel.grid(row=2,column=0)
        thirdObjectNameCombobox.grid(row=2,column=1)
        fourthObjectNameLabel.grid(row=0,column=2)
        fourthObjectNameCombobox.grid(row=0,column=3)
        fifthObjectNameLabel.grid(row=1,column=2)
        fifthObjectNameCombobox.grid(row=1,column=3)

        """ ALGORTIHM SETTINGS """
        objectTresholdLabel = ttk.Label(self, text="Object Treshold:")
        self.objectTresholdScale = ttk.Scale(self, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.root.objectTreshold, length=longestWorldLength*7, command=self.handleScale)
        objectTresholdInfo = ttk.Label(self, textvariable=self.root.objectTreshold) 

        nmsThresholdLabel = ttk.Label(self, text="NMS Treshold:")
        self.nmsThresholdScale = ttk.Scale(self, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.root.nmsThreshold, length=longestWorldLength*7, command=self.handleScale)
        nmsThresholdInfo = ttk.Label(self, textvariable=self.root.nmsThreshold)

        objectTresholdLabel.grid(row=0,column=4)
        self.objectTresholdScale.grid(row=1,column=4)
        objectTresholdInfo.grid(row=2,column=4)

        nmsThresholdLabel.grid(row=0,column=5)
        self.nmsThresholdScale.grid(row=1,column=5)
        nmsThresholdInfo.grid(row=2,column=5)

        """ BUTTONS """
        self.startButton = tk.Button(self, text="Start", command=self.start, state="disabled")
        self.saveAndExitButton = tk.Button(self, text="Save and Exit", command=self.saveAndExit, state="disabled")
        
        self.startButton.grid(row=3,column=0, sticky="WE", columnspan=3)
        self.saveAndExitButton.grid(row=3,column=3, sticky="WE", columnspan=3)

        for curWidget in self.winfo_children() :
            curWidget.grid_configure(padx=5, pady=5)
        
        for combobox in comboboxses :
            combobox.bind("<<ComboboxSelected>>", self.handleCombo)

    def handleScale(self, *event) :
        self.root.objectTreshold.set(float(str(self.objectTresholdScale.get())[0:4]))
        self.root.nmsThreshold.set(float(str(self.nmsThresholdScale.get())[0:4]))
        self.updateVariables()

    def handleCombo(self, event) :
        self.updateVariables()

    def updateVariables(self, *event) :
        if not (self.fifthObjectName.get() == "" or self.fourthObjectName.get() == "" or self.thirdObjectName.get() == "" or self.secondObjectName.get() == "" or self.firstObjectName.get() == "") :
            self.startButton.config(state="normal")

    def start(self) :
        
        for element in self.win

    def saveAndExit(self) :
        pass


class Visualizer(ttk.Frame) :

    def __init__(self, root, parent, *args, **kwargs) :
        super().__init__(parent, *args, **kwargs)

        _imageWidth = 320
        _imageHeight = 320

        self.root = root




app = ObjectDetection()
app.mainloop()
import cv2 as opencv
import numpy as np
import datetime

objectNameFilePath      = "Assets/Object Class Names.txt"
configurationsFilePath  = "Assets/Configurations.pbtxt"
weightsFilePath         = "Assets/Weights.pb"

objectTreshold = 0.45
nmsThreshold = 0.2

captureDevice = opencv.VideoCapture(0)

allObjectNames = [cName[0:-1] for cName in open(objectNameFilePath,'r')]

capModel = opencv.dnn_DetectionModel(weightsFilePath,configurationsFilePath)
capModel.setInputSize(320,320)
capModel.setInputScale(1.0/ 127.5)
capModel.setInputMean((127.5, 127.5, 127.5))
capModel.setInputSwapRB(True)

datas = {}

for i in range(100):
    
    capturedImage = opencv.cvtColor(captureDevice.read()[1],opencv.COLOR_BGR2RGB)

    originalImage = capturedImage.copy()

    objectIDs, confidences, bboxes = capModel.detect(capturedImage,confThreshold=objectTreshold)

    bboxes = list(bboxes)
    confidences = list(np.array(confidences).reshape(1,-1)[0])
    confidences = list(map(float,confidences))
    
    stats = opencv.dnn.NMSBoxes(bboxes,confidences,objectTreshold,nmsThreshold)

    for i in stats:
        i = int(i)
        box = bboxes[i]
        x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
        
        objectName = allObjectNames[objectIDs[i]-1].lower()
        objectStat = "OK"
        objectCoordinates = (x1,y1,x2,y2)
        date, time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S").split(" ")

        opencv.rectangle(capturedImage, (x1,y1),(x1+x2,y2+y1), color=(0, 255, 0), thickness=2)
        opencv.putText(capturedImage,objectName.upper(),(box[0]+10,box[1]+30), opencv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        datas[objectName] = [objectStat,objectCoordinates,date,time]

    # show original and processed images in a window togehter
    opencv.imshow("Original",originalImage)
    opencv.imshow("Output",capturedImage)
    opencv.waitKey(1)

opencv.destroyAllWindows()

print(datas)
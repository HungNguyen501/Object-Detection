import cv2
import numpy as np

windowName = "Live"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 150)

classNames = []
classFile = 'object_labels.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

configPath = 'ssd_mobilenet_v3_large.pbtxt'
weightspath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pb'

net = cv2.dnn_DetectionModel(weightspath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, drawBox=True, objects=[], confThreshold=0.45, nmsThreshold = 0.2):
    objecstInfo = []
    if len(objects) == 0:
        objects = classNames
    """
    bbox: bounding box
    confs: confidence
    """
    classIds, confs, bbox = net.detect(img, confThreshold = confThreshold)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, score_threshold = confThreshold, nms_threshold = nmsThreshold)
    
    for i in indices:
        i = i[0]        
        objectName = classNames[classIds[i][0]-1]
        if objectName in objects:
            box = bbox[i]
            objecstInfo.append([box, objectName])               
            if (drawBox):                   
                x,y,w,h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x, y), (x+w, h+y), color=(255, 0, 0), thickness=2)
                label = "{}: {:.2f}%".format( objectName.upper(), confs[0] * 100)
                cv2.putText(img, label, (box[0] + 0, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
    return objecstInfo            
    
if __name__ == "__main__":
    while True:
        success, img = cap.read()
        objecstInfo = getObjects(img) #objects = ['person','bottle', 'cup', 'keyboard']
        #if len(objecstInfo) > 0: print(objecstInfo)        
        cv2.imshow(windowName, img)    
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit program
            exit()    
    
    
    
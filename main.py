import cv2

thres = 0.5 #threshold to detect object

windowName = "Live"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)


classNames = []
classFile = 'object_labels.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

configPath = 'ssd_mobilenet_v3_large.pbtxt'
weightspath = 'pre_trained_model.pb'

net = cv2.dnn_DetectionModel(weightspath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold = thres)
    #print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox): 
            cv2.rectangle(img, box, color=(255, 0, 0), thickness = 2)
            label = "{}: {:.2f}%".format( classNames[classId-1], confidence * 100)
            cv2.putText(img, label, (box[0] + 0, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(windowName, img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit program
        exit()
    
    
    
    
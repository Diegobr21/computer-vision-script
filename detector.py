import cv2 as cv
import numpy as np
import wget
from os import mkdir, path
from os.path import join, abspath, dirname, exists

file_path = abspath(__file__)
file_parent_dir = dirname(file_path)
config_dir = join(file_parent_dir, 'config')
inputs_dir = join(file_parent_dir, 'inputs')
yolo_weights_path = join(config_dir, 'yolov3.weights')
yolo_names_path = join(config_dir, 'coco.names')
yolo_config_path = join(config_dir, 'yolov3.cfg')
input_image = join(inputs_dir, 'kemy.jpg')

net = cv.dnn.readNet(yolo_weights_path, yolo_config_path)
#To load all objects that have to be detected
classes=[]
with open(yolo_names_path,"r") as file_object:
    lines = file_object.readlines()

for line in lines:
    classes.append(line.strip("\n"))

#Defining layer names
layer_names = net.getLayerNames()
output_layers = []
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i[0]-1])

img = cv.imread(input_image)
height, width, channels = img.shape

#Extracting features to detect objects
blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#We need to pass the img_blob to the algorithm
net.setInput(blob)
outs = net.forward(output_layers)

#Displaying information on the screen
class_ids = []
confidences = []
boxes = []
for output in outs:
    for detection in output:
        #Detecting confidence in 3 steps
        scores = detection[5:]                #1
        class_id = np.argmax(scores)          #2
        confidence = scores[class_id]        #3

        if confidence > 0.5: #Means if the object is detected
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            #Drawing a rectangle
            x = int(center_x-w/2) # top left value
            y = int(center_y-h/2) # top left value

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            # cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#Removing Double Boxes
indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

for i in range(len(boxes)):
    if i in indexes[0]:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]  # name of the objects
       
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(img, label, (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

title = str(len(boxes))
cv.imshow(title, img)
cv.waitKey(0)
cv.destroyAllWindows()
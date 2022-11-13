#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import cv2
import os

import torch
import torchvision
from torchvision import transforms

from PIL import Image


# In[6]:


import os
#directories
#base_dir = os.path.dirname(__file__)
#face detction model:::::::::::::::::::::::::::::::::::::::::::::
prototxt_path = os.path.join( r'deploy.prototxt')
faceDetect_path = os.path.join(r'faceDetect.caffemodel')
#mask detection model:::::::::::::::::::::::::::::::::::::::::::::
maskDetect_path = os.path.join('model_mobilenet.pth')


# In[7]:


#loading face detection model and mask detection model
print("loading models")
maskDetectModel = torch.load(maskDetect_path)
maskDetectModel.eval()
faceDetectModel = cv2.dnn.readNetFromCaffe(prototxt_path, faceDetect_path)
device = torch.device("cpu")
maskDetectModel=maskDetectModel.to(device)


# In[8]:



#function to detect face
def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceDetectModel.setInput(blob)
    detections = faceDetectModel.forward()
    faces=[]
    positions=[]
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX,startY)=(max(0,startX-15),max(0,startY-15))
        (endX,endY)=(min(w-1,endX+15),min(h-1,endY+15))
        confidence = detections[0, 0, i, 2]
        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            face = frame[startY:endY, startX:endX]
            faces.append(face)
            positions.append((startX,startY,endX,endY))
    return faces,positions

#function to detect mask
def detect_mask(faces):
    predictions = []
    image_transforms = transforms.Compose([transforms.Resize(size=(64,64)),
                                           #transforms.Grayscale(1),
                                           #transforms.RandomRotation(degrees=15),
                                           #transforms.ColorJitter(),
                                           #transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                          ])
    if (len(faces)>0):
        for img in faces:
            img = Image.fromarray(img)
            img = image_transforms(img)
            img = img.unsqueeze(0)
            prediction = maskDetectModel(img)[0]
            #prediction = prediction.argmax()
            predictions.append(prediction)#.data)
    return predictions
#video streaming
cap = cv2.VideoCapture(0)
cap.set(3, 700)
cap.set(4, 520)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        #print(frame.shape)
        #frame=cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
        #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # Our operations on the frame come here
        (faces,postions) = detect_face(frame)
        predictions=detect_mask(faces)
        if len(faces)>0:
            for(box,prediction) in zip(postions,predictions):
                #print(prediction)
                (startX, startY, endX, endY) = box
                ###################################################
                mask_p=float(format(float(prediction[0]),'.3f'))
                mask='Mask: '+str(mask_p)
                ##################################################
                no_mask_p=float(format(float(prediction[1]),'.3f'))
                no_mask=' No Mask: '+str(no_mask_p)
                ########################################################
                label=mask+no_mask
                #print(mask_p,no_mask_p)
                ############################################################
                color = (0, 255, 0) if mask_p>=0.70 else (255,0,0)
                #label = "Mask" if prediction == 0 else "No Mask"
                #color = (0, 255, 0) if label == "Mask" else (255,0,0)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.rectangle(frame,(startX, startY),(endX, endY),color,2)
        # Display the resulting frame
        #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.imshow('frame',frame)
        #out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
#out.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





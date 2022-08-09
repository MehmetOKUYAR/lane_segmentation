
import cv2
from tensorflow.keras.models import load_model
import time
import pandas as pd
import numpy as np


model = load_model("Desktop/line_segmentation/Unet_weight/line_early_unet.h5", compile=False)


cap = cv2.VideoCapture("video/yarisma.mp4")
if cap.isOpened() == 0:
    exit(-1)

while True :
    retval, frame = cap.read()
    frame=cv2.resize(frame,(512,512))
    img = frame.copy()
    img = img/255
    img = np.expand_dims(img,axis=0)
    
    pred = model.predict(img)
    
    pred = pred.reshape(512,512,1)
    pred = pred.astype(np.uint8)


    h, w, d = pred.shape
    image=frame.copy()
    
    mask0=pred.copy()
    mask6=pred.copy()
    #segmentasyon=mask.copy()
    sagserit=pred.copy()
    solserit=pred.copy()
    
    #search_topseg = 2*h/3
    #segmentasyon[0:search_topseg, 0:w] = 0


    # image roi in front of the camera
    search_top0 = h//2 
    search_bot0 = h//2 + 100
    mask0[0:search_top0, 0:w] = 0
    mask0[search_bot0:h, 0:w] = 0
    mask_sol_karsi =mask0.copy()
    mask_sag_karsi =mask0.copy()

    sol_roi_x= 2*w//5
    sol_roi_y= 3*h//4 

    sag_roi_x=3*w//5
    sag_roi_y=3*h//4

    
    mask_sol_karsi[:,sol_roi_x:]=0
    mask_sag_karsi[:,:sag_roi_x]=0
    
    search = h//2 + 150
    mask6[0:search, 0:w] = 0

    # image roi right of the camera
    sagserit[:sag_roi_y, :] = 0
    sagserit[sag_roi_y:, :sag_roi_x] = 0


    # image roi left on the camera
    solserit[:sol_roi_y, :] = 0
    solserit[sol_roi_y:, sol_roi_x:] = 0

    # center line detections
    M = cv2.moments(solserit) 
    N = cv2.moments(sagserit)

    SagM = cv2.moments(mask_sag_karsi)
    SolM = cv2.moments(mask_sol_karsi)

    S = cv2.moments(mask0)
    
    # lane detection imshow
    pred *=255
    pred = pred.astype(np.uint8)
    red=np.zeros((image.shape[0],image.shape[1],image.shape[2]),np.uint8)
    cv2.rectangle(red,(0,0),(red.shape[1],red.shape[0]),(0,0,255),-1)
    maskbgr= cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    redmask=cv2.bitwise_and(maskbgr,red)
    # serit silme
    bitnot=cv2.bitwise_not(pred)
    bitnot3= cv2.cvtColor(bitnot, cv2.COLOR_GRAY2BGR)
    bitw=cv2.bitwise_and(image,bitnot3)
    image=cv2.add(bitw,redmask)

    try :

        cx1 = int(M['m10']/M['m00'])
        cy1 = int(M['m01']/M['m00'])
        cv2.circle(image,(cx1,cy1),5,(213,164,0),-1)
        cv2.putText(image,"sol ref",(cx1-10,cy1-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,155,0),1,cv2.LINE_4)

    except:
        print("serit tespit edilemedi")

    try:
        cx2 = int(N['m10']/N['m00'])
        cy2 = int(N['m01']/N['m00'])
        cv2.circle(image,(cx2,cy2),5,(213,164,0),-1)
        cv2.putText(image,"sag ref",(cx2-10,cy2-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,155,0),1,cv2.LINE_4)
    except:
        print("serit tespit edilemedi")
    

    image= cv2.resize(image,(1700,800))
    cv2.imshow('lane segmentation',image)
    if cv2.waitKey(1) == ord('q'):
        break





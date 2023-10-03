import os.path
import cv2
import numpy as np
from util import get_parking_spots_bbox
from util import parked_or_not
from  util import calc_diff
import  matplotlib.pyplot as plt

mask =os.path.join('../mask','mask_1920_1080.png')
video_path=os.path.join('../Video/data','parking_1920_1080_loop.mp4')

mask=cv2.imread(mask,0)

cap= cv2.VideoCapture(video_path)

#Get bounding box for parking slot location using mask
Connected_Components=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
spots =get_parking_spots_bbox(Connected_Components)

spots_status= [None for j in spots]
diffs=[None for j in spots]

previous_frame= None

frame_nur=0
step=30
ret= True
while ret:
    ret, frame = cap.read()

    if frame_nur % step == 0 and previous_frame is not None:
        for spot_indx,spot in enumerate(spots):
            x1, y1, w, h = spot
            # Crop the spot and process that fram
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_indx]=calc_diff(spot_crop,previous_frame[y1:y1 + h, x1:x1 + w, :])

        #print([diffs[j] for j in np.argsort(diffs)][::-1])
        #plt.hist([diffs[j]/np.max(diffs) for j in np.argsort(diffs)][::-1])
        #plt.show()
    #Make prediction every 30 frame or 5 sec : For optimizing sol.
    if frame_nur % step == 0:
        #for spot_indx,spot in enumerate(spots):
        if previous_frame is None:
            arr_ =range(len(spots))
        else:
            arr_=[j for j in np.argsort(diffs) if diffs[j]/np.max(diffs)>0.4 ]
        # Make prediction ,if the previous frame is different than the current frame
        for spot_indx in arr_:
            spot =spots[spot_indx]
            x1,y1,w,h=spot
            #Crop the spot and process that frame
            spot_crop=frame[y1:y1+h,x1:x1+w,:]

            #Predict the Spot is available or not
            spot_status=parked_or_not(spot_crop)
            spots_status[spot_indx]= spot_status

    if frame_nur % step == 0:
        previous_frame= frame.copy()

    for spot_indx, spot in enumerate(spots):
        spot_status=spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]
        #if spot_status if True means parking spot is empty else not empty
        if spot_status :
            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2) #Green box for empty spot
        else:
            cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2) #Red box for full parking spot
    cv2.rectangle(frame,(80,20),(650,80),(0,0,0),-1)
    cv2.putText(frame,"Available Parking Spots: {}/{}".format(str(sum(spots_status)),str(len(spots_status))),(100,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    if cv2.waitKey(25)& 0XFF == ord('q'):
        break

    frame_nur +=1



cap.release()
cv2.destroyAllWindows()
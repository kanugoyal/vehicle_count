from itertools import count
import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

min_width_rec = 80    # min width of rectangle
min_heig_rec = 80     #min height of rectangle


countLine_posi = 550

# line to subtract background 
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# function to detect vehicle
def center_handle(x,y,w,h):
    x1= int(w/2)
    y1= int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy  

detect = []
offset = 6
counter = 0


while True:
    ret,frame1 = cap.read()
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3),5)

    # to apply on each frame
    imgsub = algo.apply(blur)
    dlate = cv2.dilate(imgsub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    dlatedata = cv2.morphologyEx(dlate,cv2.MORPH_CLOSE, kernel)
    dlatedata = cv2.morphologyEx(dlatedata,cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dlatedata, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # cv2.imshow("Detect",  dlatedata)

    cv2.line(frame1, (25, countLine_posi), (1200,countLine_posi), (255,127,0), 5)

    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        valid_counter = (w>= min_width_rec) and (h>= min_heig_rec)
        if not valid_counter:
            continue

        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,130),4)

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0,130,255),3)

        for(x,y) in detect:
            if y<(countLine_posi + offset) and  y>(countLine_posi - offset):
                counter +=1
            cv2.line(frame1,(25,countLine_posi),(1200, countLine_posi), (0,150,250),4)
            detect.remove((x,y))
            print("vehicle count :"+ str(counter))

    cv2.putText(frame1, "Vehicle COUNT :" +str(counter),(450, 70),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,250),5)


    cv2.imshow("original", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
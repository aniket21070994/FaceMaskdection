import cv2 as cv
import numpy as np
cap=cv.VideoCapture(0);
eye_casscade=cv.CascadeClassifier('haarcascade_eye.xml')
face_casscade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')

lab="recognization under process"
print("")
while cap.isOpened():
    ret,frame=cap.read()
    if ret:
        ct=cv.resize(frame,(500,500))
        try:  
            face=face_casscade.detectMultiScale(ct,1.3,5)
           
            for (x,y,w,h) in face:
                (x,y,w,h)=face[0]
                ct=cv.rectangle(ct,(x,y),(x+w,y+h),(255,0,0),3)
                mface=ct[y:y+h,x:x+w]
                lab="gating face"
                try:
                    news=[]
                    data=mface[y:y+h,x:x+w]
                    news.append(cv.resize(mface,(70,70)))
                    news=np.array(news)
                    pnn=keras.models.load_model(r"D:\farmhelp\mask_model.h5")
                    p=pnn.predict(news)
                    g=np.argmax(p)
                    lables=["No mask","mask"]
                    print(lables[g])
                    cv.putText(ct,lables[g], (10,450), font, 1, (255, 255, 0), 4, cv.LINE_AA)
                except:
                     lab="not predicting"
#                       
               
        except:
            pass
                
        try:
            eye=eye_casscade.detectMultiScale(ct,1.3,5)
                       
            for (ex,ey,ew,eh) in eye:
                ct=cv.rectangle(ct,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
               

        except:
            lab="not predicting_2"
        
#         font = cv.FONT_HERSHEY_SIMPLEX
#         cv.putText(ct,lab, (10,450), font, 1, (255, 255, 0), 4, cv.LINE_AA)
        cv.imshow('data',ct)
      
            
        if cv.waitKey(50) & 0xff==ord('z'):
            break;
    else:
        break;
cap.release()
cv.destroyAllWindows()

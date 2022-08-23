import os
import cv2
import numpy as np
import face_recognition

path ='data1'

images=[]

classNames=[]

mylist=os.listdir(path)


# print(mylist)

for cl in mylist:
    currentimage=cv2.imread(f'{path}/{cl}')
    images.append(currentimage)
    classNames.append(os.path.splitext(cl)[0])

# print(mylist)
# print(classNames)


def findencodings(images):
    encodelist=[]

    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknownfaces=findencodings(images)

# print(encodelistknownfaces)
print("encoding completed")


cap=cv2.VideoCapture(0)

while True:
    sucess,img=cap.read()
    imagesmall=cv2.resize(img,(0,0),None,0.25,0.25)


    face_in_frame=face_recognition.face_locations(imagesmall)
    encoded_face=face_recognition.face_encodings(imagesmall,face_in_frame)

    # print(face_in_frame)

    for encodeface,faceloc in zip(encoded_face,face_in_frame):
        matches=face_recognition.compare_faces(encodelistknownfaces,encodeface)
        facedistance=face_recognition.face_distance(encodelistknownfaces,encodeface)

        # print(facedistance)
        matchindex=np.argmin(facedistance)

        if matches[matchindex]:
            name=classNames[matchindex]
            print(name)

            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)



    cv2.imshow('recognition',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



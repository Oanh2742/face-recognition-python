import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

IMG_PATH = "./data/base_images/"
count = 25
leap = 1

usr_name = input("Enter your name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)


mtcnn = MTCNN(margin = 20, keep_all=False, select_largest = True, post_process=False, device = device)
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while video.isOpened() and count:
    ret, frame = video.read()
    if mtcnn(frame) is not None and leap%3:
        t = datetime.now()
        path = str(USR_PATH+'/{}_{}.jpg'.format(t.strftime("%d-%m-%y_%H-%M"),str(count).zfill(2)))
        face_img = mtcnn(frame, save_path = path)
        count-=1
    leap+=1
    cv2.imshow('Face Capturing', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
video.release()
cv2.destroyAllWindows()
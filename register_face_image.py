import cv2
import torch
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face
import numpy as np
from PIL import Image
from datetime import datetime

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
image_name = input("Enter image file name: ")
img = cv2.imread("./raw/" + image_name)
img2 = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
count = 0
boxes, probs, points = mtcnn.detect(img,landmarks=True)
if boxes is not None:
    for box in boxes:
        count = count + 1
        label = "{}?".format(count)
        box = box.astype(int)
        img2 = cv2.rectangle(img2,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        img2 = cv2.rectangle(img2, (box[0], box[3]), (box[0] + w, box[3]+18), (0,255,0), -1)
        img2 = cv2.putText(img2, label, (box[0], box[3]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    image = Image.fromarray(img2)
    image.show()
    count = 0
    for box in boxes:
        count = count + 1
        usr_name = input("Person {} is: ".format(count))
        if usr_name=="unknown" or usr_name=="u":
            continue
        t = datetime.now()
        path = "./data/base_images/{}/{}_{}.jpg".format(usr_name,t.strftime("%d-%m-%y_%H-%M"),str(count).zfill(2))
        extract_face(img,box,margin=20,save_path=path)
    
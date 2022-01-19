import glob
import torch 
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face
import os
import cv2
from datetime import datetime

SRC_PATH = "./raw/CASIA-WebFace"

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

count_people = 0
count_imgs = 0
IMGS_DOWN = 3
IMGS_UP = 8
PEOPLE_DOWN = 100
PEOPLE_UP = 200

for usr in os.listdir(SRC_PATH):
    count_people = count_people + 1
    if count_people <= PEOPLE_DOWN:
        continue
    if count_people > PEOPLE_UP:
        break
    count_imgs = 0
    for file in glob.glob(os.path.join(SRC_PATH, usr)+"/*.jpg"):
        count_imgs = count_imgs +1
        if count_imgs <= IMGS_DOWN:
            continue
        if count_imgs > IMGS_UP:
            break
        img = cv2.imread(file)
        boxes, probs, points = mtcnn.detect(img,landmarks=True)
        if boxes is not None:
            for box in boxes:
                t = datetime.now()
                path = "./data/base_images/{}/{}_{}.jpg".format(usr,t.strftime("%d-%m-%y_%H-%M"),str(count_imgs).zfill(3))
                extract_face(img,box,margin=20,save_path=path)
print("Done")
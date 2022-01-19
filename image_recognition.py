import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import os

power = pow(10,6)
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)
model.eval()

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)

def load_faceslist():
    embeds = torch.load("./data/known_faces.pth")
    names = np.load("./data/known_names.npy")
    return embeds, names
def extract_face(box, img, margin=20):
    face_size = 160
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin)] #tạo margin bao quanh box cũ
    new_box = [int(max(box[0] - margin[0] / 2, 0)),
                int(max(box[1] - margin[1] / 2, 0)),
                int(min(box[2] + margin[0] / 2, w)),
                int(min(box[3] + margin[1] / 2, h))]
    img = img[new_box[1]:new_box[3], new_box[0]:new_box[2]]
    face = cv2.resize(img,(160, 160), interpolation=cv2.INTER_AREA).copy()
    face = Image.fromarray(face)
    return face
def inference(model, face, local_embeds, threshold = 0.8):
    #local: [n,512] voi n la so nguoi trong known_faces
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds) #[1,512]
    # print(detect_embeds.shape)
                    #[1,512,1]                                      [1,512,n]
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    # print(norm_diff)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi
    
    min_dist, embed_idx = torch.min(norm_score, dim = 1)
    #print(min_dist*power, names[embed_idx])
    # print(min_dist.shape)
    min_dist = min_dist.double()*power
    if min_dist > threshold:
        return -1,torch.tensor([-1])
    else:
        return embed_idx, min_dist

current_path = os.path.abspath(".")
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL) 
image_name = input("Enter image file name: ")
frame = cv2.imread(current_path + "/raw/" + image_name)
w ,h = frame.shape[0:2]

embeddings, names = load_faceslist()

boxes, probs = mtcnn.detect(frame)
if boxes is not None:
    for box in boxes:
        box = box.astype(int)
        face = extract_face(box,frame)
        idx, score = inference(model, face, embeddings)
        score = score.item()
        label = "unknown"
        if idx != -1:
            label = names[idx] + "_{:.4f}".format(score)
        #draw sth in frame
        frame = cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(0,255,0),4)
        (wt, ht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        frame = cv2.rectangle(frame, (box[0], box[3]), (box[0] + wt, box[3] + ht), (0,255,0), -1)
        frame = cv2.putText(frame, label, (box[0], box[3] + ht), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
# Display the resulting frame
cv2.imshow('Frame',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
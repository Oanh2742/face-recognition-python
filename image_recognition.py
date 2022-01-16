import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time

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

def inference(model, face, local_embeds, threshold = 3
):
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
    print(min_dist*power, names[embed_idx])
    # print(min_dist.shape)
    if min_dist*power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

def extract_face(box, img, margin=20):
    face_size = 160
    img_size = img.shape[0:2]
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ] #tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face

if __name__ == "__main__":
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))
    power = pow(10, 6)
    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model.eval()
    mtcnn = MTCNN(thresholds= [0.6, 0.7, 0.7] ,keep_all=True, device = device)
    embeddings, names = load_faceslist()
    
    image_name = input("Enter image file name: ")
    img = cv2.imread("./raw/" + image_name)
    w = img.shape[0]
    h = img.shape[1]

    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        for box in boxes:
            box = box.astype(int)
            face = extract_face(box, img)
            idx, score = inference(model, face, embeddings)
            label = "unknown"
            if idx != -1:
                label = names[idx]
            img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),4)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            img = cv2.rectangle(img, (box[0], box[3]), (box[0] + w, box[3]+20), (0,255,0), -1)
            img = cv2.putText(img, label, (box[0], box[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    image = Image.fromarray(img)
    image.show()
import glob
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np

IMG_PATH = "./data/base_images"
DATA_PATH = "./data"

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)
    
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

model.eval()

embeddings = []
names = []
count = 0

for usr in os.listdir(IMG_PATH):
    count = count + 1
    for file in glob.glob(os.path.join(IMG_PATH, usr)+"/*.jpg"):
        # print(usr)
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            embedding=model(trans(img).to(device).unsqueeze(0))
            embeddings.append(embedding)
            names.append(usr)
    
embeddings = torch.cat(embeddings)
names = np.array(names)

torch.save(embeddings, DATA_PATH+"/known_faces.pth")
np.save(DATA_PATH+"/known_names", names)
print('Update Completed! There are {0} people in FaceLists'.format(count))
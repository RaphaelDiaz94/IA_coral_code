import cv2
import pathlib
from pathlib import Path
import glob
pain = glob.glob("/home/raph/Bureau/VisDrone2019-DET-train/images_new/*.jpg")
for i in range(len(pain)) :

    path = pain[i]
    filename = pathlib.Path(path).stem
    image = cv2.imread(path)
    h=100
    w=200
    w,h = image.shape[:-1]
    diff = h-w
    crop = image[:, int(diff/2):int(w+diff/2)]
    crop.shape
    cv2.imwrite("/home/raph/Bureau/VisDrone2019-DET-train/images_new/"+filename+".jpg",crop)

for i in range(len(pain)) :

    path = pain[i]
    filename = pathlib.Path(path).stem
    image = cv2.imread(path)
    w,h = image.shape[:-1]


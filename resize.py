import cv2
import pathlib
from pathlib import Path
import glob
pain = glob("/home/mendel/IA_coral_code/image/*.jpg")
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
    cv2.imwrite("/home/mendel/IA_coral_code/image/"+filename+".jpg",crop)
print("fin")
for i in range(len(pain)) :

    path = pain[i]
    filename = pathlib.Path(path).stem
    image = cv2.imread(path)
    w,h = image.shape[:-1]
    print("w:",w,"h:",h)

print("fin")

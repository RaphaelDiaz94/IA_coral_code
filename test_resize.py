import cv2
import pathlib
from pathlib import Path
import glob


pain = glob.glob("/home/raph/Bureau/VisDrone2019-DET-train/images_new/*.jpg")
for i in range(len(pain)) :

    path = pain[i]
    filename = pathlib.Path(path).stem
    image = cv2.imread(path)
    res = cv2.resize(image, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/home/raph/Bureau/VisDrone2019-DET-train/images_new/"+filename+".jpg",res)

print("fin")
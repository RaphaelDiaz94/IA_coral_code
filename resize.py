from pathlib import Path
[16:01]
pain = glob("C:\Users\diazy\OneDrive\YNOV\2021-2022\IA_EMB\projet_fil_rouge\VisDrone2019-DET-train\VisDrone2019-DET-train\images\*.jpg")
[16:01]
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
    cv2.imwrite("C:\Users\diazy\OneDrive\YNOV\2021-2022\IA_EMB\projet_fil_rouge\VisDrone2019-DET-train\images\"+filename+".jpg",crop)
print("fin")
[16:07]
for i in range(len(pain)) :

    path = pain[i]
    filename = pathlib.Path(path).stem
    image = cv2.imread(path)
    w,h = image.shape[:-1]
    print("w:",w,"h:",h)

print("fin")

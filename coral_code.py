import argparse
import time

import os

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter




def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""

 

  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


def main():
  i=0
 
  path = '/home/mendel/CHEH_IA/im'
  print(path)
  files = os.listdir(path)
  for name in files:
    
    nb_personne = 0 
    nb_car = 0
    nb_motorcycle = 0
    nb_truck = 0

    
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects')
    args = parser.parse_args()

    labels = read_label_file('/home/mendel/CHEH_IA/coco_labels.txt') if "/home/mendel/CHEH_IA/coco_labels.txt" else {}
    interpreter = make_interpreter('/home/mendel/CHEH_IA/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
    interpreter.allocate_tensors()

    image = Image.open('/home/mendel/CHEH_IA/im/'+name)
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes',
          'loading the model into Edge TPU memory.')
    for _ in range(args.count):
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      objs = detect.get_objects(interpreter,args.threshold, scale)
      print('%.2f ms' % (inference_time * 1000))

    print('-------RESULTS--------')
    if not objs:
      print('No objects detected')

    for obj in objs:
      print(labels.get(obj.id, obj.id))
      print('  id:    ', obj.id)
      if obj.id == 0 :
        nb_personne = nb_personne+1
        
      if obj.id == 2 :
        nb_car =  nb_car+1
      if obj.id == 3:
        nb_motorcyle = nb_motorcycle+1
      if obj.id == 7:
        nb_truck = nb_truck+1
      
      print('  score: ', obj.score)
      print('  bbox:  ', obj.bbox)
    print("Nombre de personne sur l'image:",nb_personne)
    print("Nombre de voiture sur l'image:",nb_car)
    print("Nombre de moto sur l'image :",nb_motorcycle)
    print("Nombre de camion sur l'image:",nb_truck)

    
    image = image.convert('RGB')
    draw_objects(ImageDraw.Draw(image), objs, labels)
    image.save("/home/mendel/CHEH_IA/"+name)
    image.show()
    i=i+1

if __name__ == '__main__':
  print("main")
  main()
import argparse
import time

import os

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import boto3
from werkzeug.utils import secure_filename
import psycopg2
import datetime


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""

    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline="red")
        draw.text(
            (bbox.xmin + 10, bbox.ymin + 10),
            "%s\n%.2f" % (labels.get(obj.id, obj.id), obj.score),
            fill="red",
        )


def main():
    i = 0

    path = "/home/mendel/IA_coral_code/image"
    print(path)
    files = os.listdir(path)
    for name in files:

        nb_personne = 0
        nb_car = 0
        nb_motorcycle = 0
        nb_truck = 0

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "-c",
            "--count",
            type=int,
            default=5,
            help="Number of times to run inference",
        )
        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=0.4,
            help="Score threshold for detected objects",
        )
        args = parser.parse_args()

        labels = (
            read_label_file("/home/mendel/IA_coral_code/modele/labels.txt")
            if "/home/mendel/IA_coral_code/modele/labels.txt"
            else {}
        )
        interpreter = make_interpreter(
            "/home/mendel/IA_coral_code/modele/efficientdet-lite1-flo_3.tflite"
        )
        interpreter.allocate_tensors()

        image = Image.open("/home/mendel/IA_coral_code/image/" + name)
        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS)
        )

        print("----INFERENCE TIME----")
        print(
            "Note: The first inference is slow because it includes",
            "loading the model into Edge TPU memory.",
        )
        for _ in range(args.count):
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_objects(interpreter, args.threshold, scale)
            print("%.2f ms" % (inference_time * 1000))

        print("-------RESULTS--------")
        if not objs:
            print("No objects detected")

        for obj in objs:
            print(labels.get(obj.id, obj.id))
            print("  id:    ", obj.id)
            if obj.id == 0:
                nb_personne = nb_personne + 1

            if obj.id == 2:
                nb_car = nb_car + 1
            if obj.id == 3:
                nb_motorcycle = nb_motorcycle + 1
            if obj.id == 7:
                nb_truck = nb_truck + 1

            print("  score: ", obj.score)
            print("  bbox:  ", obj.bbox)
        print("Nombre de personne sur l'image:", nb_personne)
        print("Nombre de voiture sur l'image:", nb_car)
        print("Nombre de moto sur l'image :", nb_motorcycle)
        print("Nombre de camion sur l'image:", nb_truck)

        mydate = datetime.datetime.today()

        if type(nb_personne) == int:
            nomelement = "Humain"
            conn = psycopg2.connect(
                "postgres://owshwcafnfsgsx:2b4cf5ade3fb7b2f25e3f1b66cd29d5a7e420fdd1d51b4c01df4b6086f1db630@ec2-18-214-35-70.compute-1.amazonaws.com:5432/d5arg29ce13853",
                sslmode="require",
            )
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO stats (nom_element, nb_element, precision , nom_fichier) VALUES (%s, %s, %s,%s)",
                (nomelement, nb_personne, mydate, name),
            )
            conn.commit()
            cur.close()
            conn.close()

        if type(nb_car) == int:
            nomelement = "Voiture"
            conn = psycopg2.connect(
                "postgres://owshwcafnfsgsx:2b4cf5ade3fb7b2f25e3f1b66cd29d5a7e420fdd1d51b4c01df4b6086f1db630@ec2-18-214-35-70.compute-1.amazonaws.com:5432/d5arg29ce13853",
                sslmode="require",
            )
            cur = conn.cursor()
            print("connexion OK")
            cur.execute(
                "INSERT INTO stats (nom_element, nb_element, precision , nom_fichier) VALUES (%s, %s, %s,%s)",
                (nomelement, nb_car, mydate, name),
            )
            conn.commit()
            cur.close()
            conn.close()

        if type(nb_motorcycle) == int:
            nomelement = "Deux roues"
            conn = psycopg2.connect(
                "postgres://owshwcafnfsgsx:2b4cf5ade3fb7b2f25e3f1b66cd29d5a7e420fdd1d51b4c01df4b6086f1db630@ec2-18-214-35-70.compute-1.amazonaws.com:5432/d5arg29ce13853",
                sslmode="require",
            )
            cur = conn.cursor()
            print("connexion OK")
            cur.execute(
                "INSERT INTO stats (nom_element, nb_element, precision , nom_fichier) VALUES (%s, %s, %s,%s)",
                (nomelement, nb_motorcycle, mydate, name),
            )
            conn.commit()
            cur.close()
            conn.close()

        if type(nb_truck) == int:
            nomelement = "Camion"
            conn = psycopg2.connect(
                "postgres://owshwcafnfsgsx:2b4cf5ade3fb7b2f25e3f1b66cd29d5a7e420fdd1d51b4c01df4b6086f1db630@ec2-18-214-35-70.compute-1.amazonaws.com:5432/d5arg29ce13853",
                sslmode="require",
            )
            print("connexion OK")
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO stats (nom_element, nb_element, precision , nom_fichier) VALUES (%s, %s, %s,%s)",
                (nomelement, nb_truck, mydate, name),
            )
            conn.commit()
            cur.close()
            conn.close()

        image = image.convert("RGB")
        draw_objects(ImageDraw.Draw(image), objs, labels)
        image.save("/home/mendel/IA_coral_code/" + name)
        image.show()
        i = i + 1

        s3 = boto3.client(
            "s3",
            aws_access_key_id="AKIAVA5FQS27ZQZRC4ER",
            aws_secret_access_key="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",
        )
        BUCKET_NAME = "myphotobucketraph"
        img = Image.open("/home/mendel/IA_coral_code/" + name)
        if img:
            filename = secure_filename(img.filename)
            img.save(filename)
            s3.upload_file(Bucket=BUCKET_NAME, Filename=filename, Key=filename)

        msg = "Upload Done ! "
        print(msg)


if __name__ == "__main__":
    print("main")
    main()

Document de mise en service de la carte embarquée permettant la détection d‘objet dans une rue.

Data Sheet de la Google Coral :

[Coral-Dev-Board-datasheet.pdf](https://github.com/RaphaelDiaz94/IA_coral_code/files/9496906/Coral-Dev-Board-datasheet.pdf)

La carte : 

Le Coral Dev Board est un ordinateur monocarte qui contient un coprocesseur Edge TPU. Il est idéal pour le prototypage de nouveaux projets qui exigent une inférence rapide sur l'appareil pour les modèles d'apprentissage automatique. Cette page est votre guide pour commencer.

La configuration nécessite de flasher Mendel Linux sur la carte, puis d'accéder au terminal shell de la carte. Voir la documention de la carte pour sa mise en service et configuration : https://coral.ai/docs/dev-board/get-started

Afin de permettre la détéction et l'envoie des informations sur l'IHM, merci de cloner le répertoire IA_coral_code sur la carte (via SSH, cf la documentation, ou via MDT SHELL). 

Ce situer dans le répertoire du projet sur la carte : "cd home/mendel/IA_coral_code/" puis il faut démarrer le script permettant la détéction (python3 coral_code.py)

Une fois cela, on peut se déconnecter de la connexion via SSH ou MDT Shell sans éteindre la carte.

Le modèle de détection : 

Utilisation du modèle de détection : EfficientDet-Lite3 (https://coral.ai/models/object-detection/) 

Ce modèle a été réentrainé via le programme notebook : https://github.com/RaphaelDiaz94/IA_coral_code/blob/main/Code_entrainement_modele_detection.ipynb

Merci de bien utiliser le script "https://github.com/RaphaelDiaz94/IA_coral_code/blob/main/resize.py" puis "https://github.com/RaphaelDiaz94/IA_coral_code/blob/main/converter.py" avant de lancer le réentrainement du modèle. 

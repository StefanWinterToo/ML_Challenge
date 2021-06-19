import numpy as np
import os
import cv2
import sys
import pickle
import csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from models import RandomForest, SuportVectorMachine

class Samples():

    def __init__(self):
        pass

    def load_images_from_folder(self, folder):
        images = []
        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        
        return images

    def hand_finder(self, images):
        hands = []
        
        if len(images) != 28:
            for image in images:    
                img_result = []
                
                edged_image = cv2.Canny(image, 100, 200)

                i = 0

                while i < 191:
                    if edged_image[0][i] + edged_image[0][i+1] + edged_image[0][i+2] + edged_image[0][i+3] + \
                    edged_image[0][i+4] + edged_image[0][i+5] + edged_image[0][i+6] + edged_image[0][i+7] + \
                    edged_image[0][i+8] + edged_image[0][i+9] == 0:
                        img_result.append(i)
                        i += 27
                    i += 1
                
                hands.append(img_result)

        else:
            edged_image = cv2.Canny(images, 100,200)

            i = 0

            while i < 191:
                if edged_image[0][i] + edged_image[0][i+1] + edged_image[0][i+2] + edged_image[0][i+3] + \
                edged_image[0][i+4] + edged_image[0][i+5] + edged_image[0][i+6] + edged_image[0][i+7] + \
                edged_image[0][i+8] + edged_image[0][i+9] == 0:
                    hands.append(i)
                    i += 27
                i += 1

        return hands
        
    def hand_locator(self, images, hands):
        all_imgs = []

        for idx, val in enumerate(hands):
            imgs = []

            for j in val:
                img = []
                
                if j <= 3:
                    if j <= 171:
                        for i in range(5):
                            hand = images[idx][:28, j+i:j+i+28]
                            img.append(hand)
                    else:
                        for i in range(5):
                            hand = np.ones(28*28).reshape(28,28)
                            img.append(hand)
                else:
                    if j <= 171:
                        for i in range(-3,2):
                            hand = images[idx][:28, j+i:j+i+28]
                            img.append(hand)
                    else: 
                        for i in range(5):
                            hand = np.ones(28*28).reshape(28,28)
                            img.append(hand)

                imgs.append(img)

            all_imgs.append(imgs)

        return all_imgs

    def predict(self, images, trained_model_path):
        predictions = []
        
        for image in images:
            img = []
            
            for hand in image:
                hands = []
                
                for diff in hand:
                    if diff.shape[1] != 28:
                        break
                    else:
                        svc = SuportVectorMachine("SVM")
                        model = pickle.load(open(trained_model_path, "rb"))
                        pred = svc.predict_data(model, diff.reshape(1,784))
                        hands.append(str(pred[0]))
                
                img.append(hands)
                
            predictions.append(img)
            
        return predictions

    def beautifier(self, predictions):
        final = [] 
        for i in predictions:
            k = []
            
            for j in range(5):
                y = []
                
                for x in range(len(i)):
                    y.append(i[x][j])
                
                for f in range(len(y)):
                    y[f] = y[f].zfill(2)
                y = "".join(y)
                
                k.append(y)
                
            final.append(k)
    
        return final

    def load_true_labels(self, folder):
        labels = []
        
        for file in os.listdir(folder):
            labels.append(file.split(".")[0])
            
        return labels

    def convertSaveArray(self, file, data):
        open(file, "a").close()
        with open(file, "w") as f:
            n = np.array(data)
            string = ""
            for entry in n:
                for i, row in enumerate(np.swapaxes(np.array(entry), 0, 1)):
                    if(i<4):
                        for k, element in enumerate(row):
                            if k >= 0:
                                string = string + element
                        string = string + ","
                    else:
                        for k, element in enumerate(row):
                            string = string + element
                        writer = csv.writer(f, delimiter = " ", escapechar=' ', quoting=csv.QUOTE_NONE)
                        writer.writerow([string])
                        string = ""
            f.close()


    def savePredcitions(self, file, data):
        open(file, "a").close()
        with open(file, "w") as f:
            for row in data:
                w = csv.writer(f, dialect="excel")
                w.writerow(row)
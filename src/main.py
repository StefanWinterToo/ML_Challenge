import numpy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.metrics import classification_report
from models import SuportVectorMachine
from termcolor import colored

def train_model(filename, X_train, y_train):
    model = classifier.train_pipeline(X_train, y_train)
    pickle.dump(model, open(filename, "wb"))

    return model

    
def print_score(classifier, model, X_test, y_test):
    accuracy, params = classifier.best_parameters(model, X_test, y_test)
    print("Accuracy of {} can be achieved with the following \
    parameters: {}".format(accuracy, params))

    # Predict Data
    y_pred = classifier.predict_data(model, X_test)
    print()
    print(classification_report(y_test, y_pred))

    # Save Predictions to save time when rerunning
    save_predictions(classifier.name, y_pred)

def save_predictions(name, predictions):
    file = "data/predictions_training/" + str(name) + ".csv"
    numpy.savetxt(file, predictions, delimiter=",")
    
def load_data():
    with numpy.load("/Users/stefanwinter/Local/data/train_data_label.npz") as data:
        train_data = data["train_data"]
        train_label = data["train_label"]
    
    with numpy.load("/Users/stefanwinter/Local/data/test_data_label.npz") as data:
        test_data = data["test_data"]
        test_label = data["test_label"]

    return train_data, train_label, test_data, test_label

if __name__ == "__main__":
    classifier = SuportVectorMachine("SVM")

    X_train, y_train, X_test, y_test = load_data()

    """ Temporarily Limit Data """
    X_train = X_train[:50,:]
    y_train = y_train[:50]
    X_test = X_test[:10, :]
    y_test = y_test[:10]
    """                        """
    
    # Check if we already have a trained model in our data folder
    filename = "data/final_model.sav"

    if Path(filename).is_file():
        print(colored("Found a trained model in: {}".format(filename), "blue"))
        model = pickle.load(open(filename, "rb"))
    else:
        print(colored("Trained model from scratch.", "red"))
        model = train_model(filename, X_train, y_train)

    print_score(classifier, model, X_test, y_test)
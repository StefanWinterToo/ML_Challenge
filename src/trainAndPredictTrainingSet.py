import numpy
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import classification_report
from models import RandomForest, SuportVectorMachine
from termcolor import colored

class TrainPredict:
# Trains classifier and saves predictions in separate file

    def __init__(self, cl, train_labels, test_labels):
        self.cl = cl
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.exec(cl)

    def train_model(self, classifier, filename, X_train, y_train):
        if classifier.name == "SVM":
            model = classifier.train_pipeline(X_train, y_train)
            pickle.dump(model, open(filename, "wb"))
        elif classifier.name == "rf":
            params_flag = input("Train rf with hyperparameters? (yes/no) ")
            model = classifier.train_pipeline(X_train, y_train, params_flag)
            pickle.dump(model, open(filename, "wb"))
        else:
            print(colored("Check the naming of the classifier!", "red"))

        return model


    def print_score(self, classifier, model, X_test, y_test):
        accuracy, params = classifier.best_parameters(model, X_test, y_test)
        print("Accuracy of {} can be achieved with the following \
        parameters: {}".format(accuracy, params))

        # Predict Data
        y_pred = classifier.predict_data(model, X_test)
        print()
        print(classification_report(y_test, y_pred))

        # Save Predictions to save time when rerunning
        self.save_predictions(classifier.name, y_pred)


    def save_predictions(self, name, predictions):
        file = "data/predictions_training/" + str(name) + ".csv"
        numpy.savetxt(file, predictions, delimiter=",")

    # Change paths if needed!
    def load_data(self):
        with numpy.load(self.train_labels) as data:
            train_data = data["train_data"]
            train_label = data["train_label"]
        
        with numpy.load(self.test_labels) as data:
            test_data = data["test_data"]
            test_label = data["test_label"]

        return train_data, train_label, test_data, test_label


    def exec(self, cl = "SVM"):
        # Choose classifier
        if cl == "SVM":
            classifier = SuportVectorMachine(cl)
        elif cl == "rf":
            classifier = RandomForest("rf")

        X_train, y_train, X_test, y_test = self.load_data()

        """ Temporarily Limit Data """
        # X_train = X_train[:500,:]
        # y_train = y_train[:500]
        # X_test = X_test[:100, :]
        # y_test = y_test[:100]
        """                        """
        
        # Check if we already have a trained model in our data folder
        filename = "data/final_model" + classifier.name + ".sav"

        if Path(filename).is_file():
            print(colored("Found a trained model in: {}".format(filename), "blue"))
            model = pickle.load(open(filename, "rb"))
        else:
            print(colored("Trained model from scratch.", "red"))
            model = self.train_model(classifier, filename, X_train, y_train)
        
        self.print_score(classifier, model, X_test, y_test)
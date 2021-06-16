import os
import sys
from models import SuportVectorMachine

def main():

    svm = SuportVectorMachine("foo")
    X_train, y_train, X_test, y_test = load_data()
        
    
def load_data():
    import numpy

    with numpy.load("C:\Users\stefa\Documents\data\\train_data_label.npz") as data:
        train_data = data["train_data"]
        train_label = data["train_label"]
    
    with numpy.load("C:\Users\stefa\Documents\data\\train_data_label.npz") as data:
        test_data = data["test_data"]
        test_label = data["test_label"]

    return train_data, train_label, test_data, test_label

if __name__ == "__main__":
    main()
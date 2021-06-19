import time
import numpy
from matplotlib import pyplot as plt
from trainAndPredictTrainingSet import TrainPredict
from imageDetector import Samples

# Comment out if you don't want to train and predict classifier.

TRAIN_LABELS = "/Users/stefanwinter/Local/data/train_data_label.npz"
TEST_LABELS = "/Users/stefanwinter/Local/data/test_data_label.npz"
# tp = TrainPredict("SVM", TRAIN_LABELS, TEST_LABELS) # Either SVM or rf

PATH = "/Users/stefanwinter/Local/data/sample_dataset"
TRAINED_MODEL_PATH = "/Users/stefanwinter/Local/ML_Challenge/data/trained_models/svm/final_modelSVM.sav"

imgDetector = Samples()
images = imgDetector.load_images_from_folder(PATH)
hands_found = imgDetector.hand_finder(images)
all_imgs = imgDetector.hand_locator(images, hands_found)
predictions = imgDetector.predict(all_imgs, TRAINED_MODEL_PATH)
beautified = imgDetector.beautifier(predictions)
true_labels = imgDetector.load_true_labels(PATH+"/")

# Checking the predictions agains the true labels and calculating accuracy
correct = 0
incorrect = 0
for i in range(len(beautified)):
    for j in range(len(beautified[i])):
        if beautified[i][j] == true_labels[i]:
            correct += 1
            break
        else:
            incorrect += 1
print(f"Number correct: {correct} \nAccuracy: {correct/len(true_labels)}")


# Making predictions for al 10,000 images in the full data set
start = time.time()
dataset = numpy.load("/Users/stefanwinter/Local/data/test_images_task2.npy")
dataset = dataset.astype('uint8')
# dataset = dataset[:10]
print(dataset.shape)
hands_found_dataset = imgDetector.hand_finder(dataset)
all_img_dataset = imgDetector.hand_locator(dataset, hands_found_dataset)
dataset_preds = imgDetector.predict(all_img_dataset, TRAINED_MODEL_PATH)
dataset_final_predictions = imgDetector.beautifier(dataset_preds)

FINAL_PREDS_PATH = "/Users/stefanwinter/Local/ML_Challenge/data/predictions_all_images/preds.csv"
imgDetector.convertSaveArray(FINAL_PREDS_PATH, dataset_preds)
# imgDetector.convertSaveArray(FINAL_PREDS_PATH, dataset_preds)

#file = "data/predictions_all_images/preds.csv"
#numpy.savetxt(file, dataset_preds, delimiter=",")

end = time.time()
print(f"It took {round(end-start)} seconds to make predictions for all 10,000 images. \nThis is {round((end-start)/60)} minute(s).")

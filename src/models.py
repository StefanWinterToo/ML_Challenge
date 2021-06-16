from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

class SuportVectorMachine:

    def __init__(self, name):
        self.name = name
    

    def train_pipeline(self, X_train, y_train):
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("dim_red", PCA()),
                ("svc", SVC())
            ]
        )
        param_grid = {
            "dim_red__n_components": [0.90, 0.91, 0.92], 
            "svc__kernel": ["rbf", "sigmoid", "linear"], 
            "svc__C": [4, 5, 6]
        }
        CV = GridSearchCV(pipeline, param_grid, cv = 5)
        # pipeline.get_params().keys() See all available parameters
        CV.fit(X_train, y_train)
        return CV

    def best_parameters(self, CV, X_test, y_test):
        return float(CV.score(X_test, y_test)), CV.best_params_

    def predict_data(self, model, X_test):
        return model.predict(X_test)
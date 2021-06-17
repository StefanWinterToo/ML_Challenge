from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

class Classifier:
    def __init__(self, name):
        self.name = name
    
    def best_parameters(self, CV, X_test, y_test):
        try:
            return float(CV.score(X_test, y_test)), CV.best_params_
        except:
            return float(CV.score(X_test, y_test)), "None"


    def predict_data(self, model, X_test):
        return model.predict(X_test)


class SuportVectorMachine(Classifier):
    
    def train_pipeline(self, X_train, y_train):
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("dim_red", PCA()),
                ("svc", SVC())
            ]
        )
        param_grid = {
            "dim_red__n_components": [0.91], 
            "svc__kernel": ["rbf"], 
            "svc__C": [5]
        }
        CV = GridSearchCV(pipeline, param_grid, cv = 5)

        CV.fit(X_train, y_train)
        return CV


class RandomForest(Classifier):
    def train_pipeline(self, X_train, y_train, hyperparams = "no"):
        if hyperparams == "no":
            rf = RandomForestClassifier(n_jobs=-1, random_state=999).fit(X_train, y_train)
            return rf
        else:
            pipeline = Pipeline(
                [
                    ("dim_red", PCA()),
                    ("rf", RandomForestClassifier(n_jobs=-1, random_state=999))
                ]
            )
            param_grid = {
                "dim_red__n_components": [0.85],
                "rf__max_depth": [12],
                "rf__max_features": [5],
                "rf__min_samples_leaf": [1],
                "rf__min_samples_split": [2],
                "rf__n_estimators": [19]
            }
            CV = GridSearchCV(pipeline, param_grid, cv=5)

            CV.fit(X_train, y_train)
            return CV
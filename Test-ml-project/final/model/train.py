import logging
import dill
from model.preprocess import prepare_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def main():
    X, y = prepare_data()
    # X_train, X_test, y_train, y_test = prepare_data()
    # train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    numerical_features = make_column_selector(dtype_include=['int64', 'int32', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    models = [
        #LogisticRegression(solver='liblinear', class_weight = 'balanced'),
        CatBoostClassifier(verbose=500,
                           auto_class_weights='Balanced')
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        logging.info(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    best_pipe.fit(X, y)
    with open('model/classifier_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Target action prediction model',
                'author': 'Daria V',
                'version': 1,
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
                }
            }, file)

if __name__ == '__main__':
    main()
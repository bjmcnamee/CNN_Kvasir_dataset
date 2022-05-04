import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score
from numpy import savez_compressed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import cv2 as cv
from random import randint


def get_features(files, dataset, label, df):
    row_list = []
    for file in files:
        row_list.append(file)
        row_list.append(dataset)
        row_list.append(label)
        with open(file, "r") as inFile:
            lines = inFile.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                feature, values = line.split(':')
                values = values.split(',') # values string to list
                float_values = [float(item) for item in values] # values type to float
                row_list = row_list + float_values
            series_values = pd.Series(row_list, index = df.columns)
            df = df.append(series_values, ignore_index=True)
            row_list = []
    return(df)

def split_grid_search(clf, X_train, y_train, X_dev, y_dev, list):
    # Split hyperparameters into combinations
    start_time = time.time()
    from itertools import combinations
    tuned_scores=[]
    best_params=[]
    combo_tuned_scores=[] # list to record tuned model accuracy score
    combo_param_grid=[] # list to record grid parameters
    combo_best_params=[]
    for i in range(1, len(list)+1):
       for combo in combinations(list, i):
          if i==1:
              param_grid = combo[0]
          elif i==2:
              param_grid = {**combo[0],**combo[1]}
          elif i==3:
              param_grid = {**combo[0],**combo[1],**combo[2]}
          elif i==4:
              param_grid = {**combo[0],**combo[1],**combo[2],**combo[3]}
          else:
              param_grid = {**combo[0],**combo[1],**combo[2],**combo[3],**combo[4]}
          # Grid Search with CV
          y_pred, tuned_scores, best_params = grid_search_classifier(clf, X_train, y_train, X_dev, y_dev, param_grid, tuned_scores, best_params, cv_folds=5, plot=False)
          combo_tuned_scores.append(sum(tuned_scores)/len(tuned_scores))
          combo_param_grid.append(param_grid)
          combo_best_params.append(best_params)
    df_scores = pd.DataFrame({'Tuned Scores':combo_tuned_scores, 'Parameter Grid':combo_param_grid, 'Best Parameters':combo_best_params})
    print('Done')
    print("--- %s seconds ---" % (int(time.time() - start_time)))
    return (df_scores)

def grid_search_classifier(clf, X_train, y_train, X_dev, y_dev, param_grid, tuned_scores, best_params, cv_folds, plot):
    start_time = time.time()
    grid = GridSearchCV(clf, param_grid, cv=cv_folds, scoring='accuracy')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_dev)  # unseen data
    best_param = grid.best_params_
    best_params.append(best_param)
    tuned_score = round(grid.score(X_dev, y_dev), 4)
    tuned_scores.append(tuned_score)
    print('Score :', tuned_score, '- Best Parameters :', best_param, )
    if plot == True:
        # Plot score vs all parameter values
        df_params = pd.DataFrame(grid.cv_results_['params'])
        df_scores = pd.DataFrame(grid.cv_results_['mean_test_score'], columns=['mean test score'])
        df_params = pd.merge(df_params, df_scores, left_index=True, right_index=True)
        var_names = list(df_params.columns)
        print(var_names)
        keys = len(param_grid)
        xlabel = var_names[0]
        scores = df_params.iloc[:, -1].tolist()
        plt.figure(figsize=(10, 10))
        if keys == 1:  # only one parameter --> simple plot x vs y
            params_0 = list(set(df_params.iloc[:, 0]))
            plt.plot(params_0, scores)
        else:
            params_0 = set(df_params.iloc[:, 0])
            params_1 = set(df_params.iloc[:, 1])
            text_list = not (all([str(item).isdigit() for item in params_0]))
            short_list = len(params_0) < len(params_1)
            if (text_list or short_list):
                params = params_0
                xlabel = var_names[1]
                var_legend = var_names[0]
                var_x = var_names[1]
            else:
                params = params_1
                var_param = var_names[1]
                var_x = var_names[0]
                var_legend = var_names[1]
            for param in params:
                subset = df_params.loc[df_params[var_legend] == param]
                scores = subset[var_names[-1]]
                x = subset[var_x]
                plt.plot(x, scores, label=param)
            plt.legend()
        plt.title('Accuracy vs ' + xlabel + ' Value')
        plt.xlabel('Parameter : ' + xlabel)
        plt.ylabel('Accuracy')
        clf_name = 'Tuning_' + type(clf).__name__ + '.png'
        plt.savefig(os.path.join('reports/',clf_name), dpi=300, format='png', bbox_inches='tight')
        print("--- %s seconds ---" % (int(time.time() - start_time)))
        plt.show()
    return (y_pred, tuned_scores, best_params)

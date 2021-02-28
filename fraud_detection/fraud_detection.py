import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':

    ### LOAD DATA ###
    data = pd.read_csv("fraud_detection/creditcard.csv")
    data.head()

    ### SCALE DATA ###
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)
    data.head()

    X = data.iloc[:, data.columns != 'Class']
    y = data.iloc[:, data.columns == 'Class']

    # Number of data points in the minority class
    number_records_fraud = len(data[data.Class == 1])
    fraud_indices = np.array(data[data.Class == 1].index)

    # Picking the indices of the normal classes
    normal_indices = data[data.Class == 0].index

    # Out of the indices we picked, randomly select "x" number (number_records_fraud)
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)

    # Appending the 2 indices
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])


    ### UNDERSAMPLING ###
    under_sample_data = data.iloc[under_sample_indices, :]

    X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
    y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

    # Showing ratio
    print("Percentage of normal transactions: ",
          len(under_sample_data[under_sample_data.Class == 0]) / float(len(under_sample_data)))
    print("Percentage of fraud transactions: ",
          len(under_sample_data[under_sample_data.Class == 1]) / float(len(under_sample_data)))
    print("Total number of transactions in resampled data: ", len(under_sample_data))


    ### TEST TRAIN SPLIT ###
    # Whole dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print("Number transactions train dataset: ", len(X_train))
    print("Number transactions test dataset: ", len(X_test))
    print("Total number of transactions: ", len(X_train) + len(X_test))

    # Undersampled dataset
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,
                                                                                                        y_undersample,
                                                                                                        test_size=0.3,
                                                                                                        random_state=0)
    print("Number transactions train dataset: ", len(X_train_undersample))
    print("Number transactions test dataset: ", len(X_test_undersample))
    print("Total number of transactions: ", len(X_train_undersample) + len(X_test_undersample))

    ### GRIDSEARCH TUNING ###
    c_param_range = [0.01, 0.1, 1, 10, 100]

    ### UNDERSAMPLED DATA ###
    clf = GridSearchCV(LogisticRegression(), {"C": c_param_range}, cv=5, scoring='recall')
    clf.fit(X_train_undersample, y_train_undersample)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    y_true, y_pred = y_test, clf.predict(X_test)
    best_c = clf.best_params_["C"]

    lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
    lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    y_pred_undersample = lr.predict(X_test_undersample.values)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()


    ### WHOLE DATA ###
    lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
    lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    y_pred = lr.predict(X_test.values)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()








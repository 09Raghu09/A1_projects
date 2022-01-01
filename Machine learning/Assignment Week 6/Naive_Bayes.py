import os
import pathlib
import sys
from shutil import rmtree
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from DeepSVR.analysis_utils.Analysis import print_accuracy_and_classification_report, predict_classes
from DeepSVR.analysis_utils.ClassifierPlots import create_roc_curve
from matplotlib.pyplot import figure
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def sum_features(feature_import, label):
    starts_with_label = feature_import.feature.str.startswith(label)
    return feature_import.importance[starts_with_label].sum()


def create_feature_importance_plot(trained_model, feature_import, title, outpath, save_fp):
    save_fp = pathlib.Path(save_fp)
    summed_importances = []
    for label in ['disease', 'reviewer']:
        summed_importances.append([label, sum_features(feature_import, label)])
    summed_importances = pd.DataFrame(summed_importances, columns=['feature', 'importance'])
    if not os.path.exists(save_fp.parent):
        save_fp.parent.mkdir(parents=True, exist_ok=True)
    feature_import.to_pickle(save_fp)
    feature_import.sort_values('importance', ascending=False, inplace=True)
    feature_import.replace({'feature': {'var': 'variant', 'ref': 'reference', 'avg': 'average', '_se_': '_single_end_',
                                        '3p': '3_prime', '_': ' '}}, regex=True, inplace=True)
    feature_import.to_csv(outpath.parent / "NB_features.csv")
    sns.barplot(y='feature', x='importance', data=feature_import.head(30), color='darkorange')
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.savefig(outpath)


def main(outpath):
    sns.set_style("white")
    sns.set_context('poster')
    # read in training data
    training_data = pd.read_pickle('DeepSVR/data/training_data_preprocessed.pkl')
    training_data.sort_index(axis=1, inplace=True)
    aml31_training = training_data[training_data.index.str.contains('fSsMNn1DZ3AIDGk=')]
    training_data = training_data[~training_data.index.str.contains('fSsMNn1DZ3AIDGk=')]
    training_data.groupby('call').size()

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # Re-label the germline calls as failed calls
    three_class = training_data.replace('g', 'f')
    three_class.sort_index(axis=1, inplace=True)
    # Show the calls associate with training data
    three_class.groupby('call').size()

    # Get labels for training data
    Y = three_class.call.values
    # Get training data as numpy array
    X = training_data.drop(['call'], axis=1).astype(float).values

    # Split the data for cross-validation
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33, random_state=seed)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    enc = preprocessing.MultiLabelBinarizer()
    Y_one_hot = enc.fit_transform(Y_train)

    estimator = GaussianNB()

    probabilities = cross_val_predict(estimator, X_train, Y_train, cv=kfold, method='predict_proba')
    report = print_accuracy_and_classification_report(Y_one_hot, predict_classes(probabilities))
    class_lookup = {0: 'Ambiguous', 1: 'Fail', 2: 'Somatic'}
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5, forward=True)
    feat_plot_title = 'Three Class Gaussian Naive Bayes Reciever Operating Characteristic Curve'
    create_roc_curve(Y_one_hot, probabilities, class_lookup, feat_plot_title, ax)

    model = estimator.fit(X_train, Y_train)
    r = permutation_importance(model, X_train, Y_train, n_repeats=30, random_state=0)
    feature_import = pd.DataFrame(columns=['feature', 'importance'])
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            feature_import = feature_import.append(
                {'feature': training_data.columns[i + 1], 'importance': f"{r.importances_mean[i]:.3f}"},
                ignore_index=True)

    figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')

    save_fp = 'DeepSVR/data/NB/feature_import.pkl'
    ftr_plot_title = 'Gaussian Naive Bayes feature importance'
    create_feature_importance_plot(model, feature_import, ftr_plot_title, outpath / "NB_feature_plot.png", save_fp)

    # Determine performance on test set
    test_prob = model.predict_proba(X_test)
    # Transform labels for predictions
    Y_test_labels = enc.fit_transform(Y_test)
    # Plot AUC for test set
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5, forward=True)
    roc_plot_title = 'Receiver Operating Characteristic - Hold out test set - Gaussian Naive Bayes'
    roc = create_roc_curve(Y_test_labels, test_prob, class_lookup, roc_plot_title, ax)
    fig.savefig(outpath / "NB_roc_plot.png")


if __name__ == '__main__':
    output_basepath = pathlib.Path("results")
    outpath = output_basepath / "Naive Bayes"

    if os.path.exists(outpath):
        print(f"Output directory {outpath} already exists. Deleting directory...")
        sleep(3)
        rmtree(outpath)

    if not os.path.exists(outpath):
        outpath.mkdir(parents=True, exist_ok=True)

    # save command line printouts to file
    stdoutOrigin = sys.stdout
    sys.stdout = open(outpath / "NB_log.txt", "w")

    main(outpath)

    sys.stdout.close()
    sys.stdout = stdoutOrigin

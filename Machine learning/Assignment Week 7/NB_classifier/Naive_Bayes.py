import itertools
import os
import sys
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate


def row_is_in_group(row, group_identifiers, group_index=0):
    """

    Parameters
    ----------
    row: Pandas dataframe row
    group_identifiers: [[str]]
    group_index: int

    Returns: boolean, Given a certain pandas dataframe row, returns whether or not the row is classified as a member of respective group.
    -------

    """
    row_id = str(row[0])
    if group_index == 1 and not any([True for subst in group_identifiers[0] if subst.lower() in row_id.lower()]):
        return True
    elif group_index == 0 and any(
            [True for subst in group_identifiers[0] if subst.lower() in row_id.lower()]):
        return True
    else:
        return any([True for substr in group_identifiers[group_index] if substr.lower() in row_id.lower()])


def plot_roc(roc_curves_dict, results_dict, title, outdir):
    """Plot roc curve and save to outdir."""
    outpath = Path(outdir) / f"{title.replace(' ', '').replace('-', '_')}.png"

    # unpack roc curves dictionary
    base_tpr = roc_curves_dict["base_tpr"]
    base_fpr = roc_curves_dict["base_fpr"]
    testing_tpr = roc_curves_dict["testing_tpr"]
    testing_fpr = roc_curves_dict["testing_fpr"]
    training_tpr = roc_curves_dict["training_tpr"]
    training_fpr = roc_curves_dict["training_fpr"]

    # unpack results dicttionary
    baseline = results_dict["baseline"]
    results = results_dict["results"]
    train_results = results_dict["train_results"]

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    baseline_label = 'Baseline - AUC: %s' % (round(baseline['auc'], 4))
    testing_label = 'Testing - AUC: %s' % (round(results['auc'], 4))
    training_label = 'Training - AUC: %s' % (round(train_results['auc'], 4))
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'grey', label=baseline_label)
    plt.plot(testing_fpr, testing_tpr, 'y', label=testing_label)
    plt.plot(training_fpr, training_tpr, 'g', label=training_label)
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('title')
    plt.savefig(outpath)
    plt.show()


def evaluate_model(predictions, probs, train_predictions, train_probs, train_labels, test_labels, outdir="results"):
    """Compare machine learning model to baseline performance.
    Computes statistics and saves ROC curve."""
    baseline = {}

    baseline['recall'] = recall_score(test_labels,
                                      [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels,
                                            [1 for _ in range(len(test_labels))])
    baseline['accuracy'] = accuracy_score(test_labels,
                                          [1 for _ in range(len(test_labels))])
    baseline['auc'] = 0.5

    results = {}

    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['accuracy'] = accuracy_score(test_labels, predictions)
    results['auc'] = roc_auc_score(test_labels, probs)

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['accuracy'] = accuracy_score(train_labels, train_predictions)
    train_results['auc'] = roc_auc_score(train_labels, train_probs)

    metrics = []
    for metric in ['recall', 'precision', 'accuracy', 'auc']:
        metrics.append((metric, round(baseline[metric], 4), round(train_results[metric], 4), round(results[metric], 4)))
    print(tabulate(metrics, headers=['Metric', 'Baseline', 'Train', 'Test']))
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    testing_fpr, testing_tpr, _ = roc_curve(test_labels, probs)
    training_fpr, training_tpr, _ = roc_curve(train_labels, train_probs)

    roc_curves_dict = {
        "base_fpr": base_fpr,
        "base_tpr": base_tpr,
        "testing_fpr": testing_fpr,
        "testing_tpr": testing_tpr,
        "training_fpr": training_fpr,
        "training_tpr": training_tpr,
    }

    results_dict = {
        "baseline": baseline,
        "results": results,
        "train_results": train_results,
    }

    plot_roc(roc_curves_dict, results_dict, title="ROC Curve - NB", outdir=outdir)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.PuRd, outpath="results"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    outpath = Path(outpath) / f"{title.replace(' ', '').replace('-', '_')}.png"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)
    plt.savefig(outpath)
    plt.show()


def sum_features(feature_import, label):
    starts_with_label = feature_import.feature.str.startswith(label)
    return feature_import.importance[starts_with_label].sum()


def create_feature_importance_plot(trained_model, feature_import, title, outdir):
    outpath = Path(outdir) / f"{title.replace(' ', '').replace('-', '_')}.png"
    summed_importances = []
    for label in ['disease', 'reviewer']:
        summed_importances.append([label, sum_features(feature_import, label)])
    summed_importances = pd.DataFrame(summed_importances, columns=['feature', 'importance'])
    feature_import.sort_values('importance', ascending=False, inplace=True)

    sns.barplot(y='feature', x='importance', data=feature_import.head(18), color='darkorange')
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.savefig(outpath)
    plt.show()

def run_feature_importance_analysis(estimator, X_test, y_test, featurelist_path):

    # Extract feature importances
    # r = permutation_importance(estimator, X_test, y_test, n_repeats=30, random_state=0)
    feature_importances_path=output_dir / "important_features.csv"
    if os.path.exists(feature_importances_path) and os.path.isfile(feature_importances_path):
        feature_import = pd.read_csv(feature_importances_path)
    # else:
    #     # feature_import = pd.DataFrame(columns=['feature', 'importance'])
    #     test_list = []
    #     for i in r.importances_mean.argsort()[::-1]:
    #         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    #             # feature_import = feature_import.append(
    #             #     {'feature': X.columns[i], 'importance': f"{r.importances_mean[i]:.3f}"},
    #             #     ignore_index=True)
    #     print(f"test_list: {test_list}")
        print(f"Writing feature importances to {feature_importances_path}")


    # prepare features_df with pymrmr.mRMR
    feature_df_analysis = pd.read_csv(featurelist_path,  delimiter="\t")
    feature_df_analysis.replace(0, y, inplace=True)
    feature_df_analysis.rename(index={0: "class"}, inplace=True)
    print(feature_df_analysis.head(n=18))
    # important_feats = pymrmr.mRMR(feature_df_analysis, 'MIQ', 18)
    # print(important_feats)

    # feature_import.to_csv(feature_importances_path, sep='\t')

    # feat_fig = px.bar(important_features, x='feature', y='importance', labels={'feature':'Features', 'importance':'Feature importance'})
    # feat_fig.write_image(output_dir / "important_features.png")

    # feat_fig = plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    # feat_plot_title = 'Gaussian Naive Bayes feature importance'
    # print("Creating Feature Importance Plot.")
    # create_feature_importance_plot(estimator, feature_import, feat_plot_title, output_dir)


def main(inpath, output_dir):
    group_identifiers = [["HD", "control"],
                         ["KRAS", "Unkown", "Liver", "Chol", "HBC", "Breast", "CRC", "Lung",
                          "Pancr", "NSCLC", "BrCa"]]

    sns.set_style("white")
    sns.set_context('poster')

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    data = pd.read_csv(inpath, index_col=0, delimiter="\t")
    # transpose --> Genes = Features
    X = data.transpose()

    # create classification vector, 0 = healthy, 1 = cancer

    y = [0 if row_is_in_group(row, group_identifiers, group_index=0) else 1 for row in X.iterrows()]

    # Split the data for cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=1 / 3, random_state=85)
    estimator = GaussianNB().fit(X_train, y_train)

    # Training predictions (to demonstrate overfitting)
    y_pred_train = estimator.predict(X_train)
    y_pred = estimator.predict(X_test)

    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    # Model Accuracy: how often is the classifier correct?
    acc = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Testing predictions (to determine performance)
    probabilities_train = estimator.predict_proba(X_train)[:, 1]
    probabilities = estimator.predict_proba(X_test)[:, 1]

    # feature analysis
    # Dataframe of the original data with the labels and desired classification used for feature importance analysis
    # with mRMR method
    features_df = X.assign(Class=y)
    featurelist_path = output_dir / "feature_list_NB.csv"
    features_df.to_csv(featurelist_path, sep='\t')
    print(f"Feature lists saved at {featurelist_path}")

    # evaluate and create roc plot:
    evaluate_model(y_pred, probabilities, y_pred_train, probabilities_train, y_train, y_test, outdir=output_dir)

    # generate confusion matrices
    training_cm = confusion_matrix(y_train, y_pred_train)
    plot_confusion_matrix(training_cm, classes=['Healthy', 'Cancer'],
                          title='Training Confusion Matrix - NB', outpath=output_dir)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Healthy', 'Cancer'],
                          title='Testing Confusion Matrix - NB', outpath=output_dir)

    # feature importance analysis
    # run_feature_importance_analysis(estimator, X_test, y_test, features_df)


if __name__ == '__main__':
    data_dirpath = Path("../data")
    if len(sys.argv) == 1:
        input_dir = data_dirpath / "input"
        inputfilename = "GSE68086_TEP_data_matrix_preprocessed.txt"
        # inputfilename = "GSE68086_TEP_data_matrix_preprocessed_bootstrapped.txt"
        inputfilepath = input_dir / inputfilename
    else:
        inputfilepath = Path(sys.argv[1])

    output_dir = Path("results")

    output_dir = output_dir / f"{inputfilepath.stem}"

    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Deleting directory...")
        # sleep(3)
        rmtree(output_dir)

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    main(inputfilepath, output_dir)

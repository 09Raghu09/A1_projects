# Bayesian Random forest Regressor

import os
import sys
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tabulate
from hdbscan import HDBSCAN
from pymethylprocess.MethylationDataTypes import MethylationArray
from pymethylprocess.general_machine_learning import MachineLearning
from seaborn import cubehelix_palette
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import linear_model
from umap import UMAP


def reduce_plot(data, labels, legend_title, outpath=Path("results")):
    outpath = outpath / f"{legend_title}.png"
    np.random.seed(42)
    plt.figure(figsize=(8, 8))
    t_data = pd.DataFrame(PCA(n_components=2).fit_transform(data), columns=['z1', 'z2'])
    t_data[legend_title] = labels
    sns.scatterplot('z1', 'z2', hue=legend_title, cmap=cubehelix_palette(as_cmap=True), data=t_data)
    plt.savefig(outpath)
    plt.show()


def importance_plot(data, outpath=Path("results")):
    outpath = outpath / f"Importance.png"
    plt.figure(figsize=(5, 5))
    sns.barplot('sample', 'Importance', data=data)
    plt.axis('off')
    plt.savefig(outpath)
    plt.show()


def main(inpath, outpath):
    sns.set()
    np.random.seed(42)

    # ## Load Data

    train_methyl_array = MethylationArray.from_pickle(inpath / "train_methyl_array.pkl")
    val_methyl_array = MethylationArray.from_pickle(inpath / "val_methyl_array.pkl")
    test_methyl_array = MethylationArray.from_pickle(inpath / "test_methyl_array.pkl")

    # ## UMAP Embed

    umap = UMAP(n_components=100)
    umap.fit(train_methyl_array.beta)
    train_methyl_array.beta = pd.DataFrame(umap.transform(train_methyl_array.beta.values), index=train_methyl_array.return_idx())
    val_methyl_array.beta = pd.DataFrame(umap.transform(val_methyl_array.beta), index=val_methyl_array.return_idx())
    test_methyl_array.beta = pd.DataFrame(umap.transform(test_methyl_array.beta), index=test_methyl_array.return_idx())

    # # ## Cluster Training Data
    #
    # model = HDBSCAN(algorithm='best')
    # train_predicted_clusters = model.fit_predict(train_methyl_array.beta.astype(np.float64))
    # reduce_plot(train_methyl_array.beta, train_methyl_array.pheno['Age'].values, 'Age', outpath)
    # reduce_plot(train_methyl_array.beta, train_predicted_clusters, 'Cluster', outpath)
    #
    # train_methyl_array.pheno['Cluster'] = train_predicted_clusters
    # output_data = train_methyl_array.pheno.groupby('Clusterf')['Age'].agg([np.mean, len])
    # print(tabulate.tabulate(output_data, headers='keys', tablefmt="pipe"))

    y_pred = {}
    scores = {}
    model = MachineLearning(RandomForestRegressor, options={})
    model.fit(train_methyl_array, val_methyl_array, 'Age')
    y_pred['train'] = model.predict(train_methyl_array)
    y_pred['val'] = model.predict(val_methyl_array)
    y_pred['test'] = model.predict(test_methyl_array)

    output_df = pd.DataFrame({"Age": val_methyl_array.pheno.Age, "Age_pred": y_pred["val"]})
    output_df.to_csv(outpath / "age_prediction.txt", sep="\t")
    scores['train'] = r2_score(train_methyl_array.pheno['Age'], y_pred['train'])
    scores['val'] = r2_score(val_methyl_array.pheno['Age'], y_pred['val'])
    scores['test'] = r2_score(test_methyl_array.pheno['Age'], y_pred['test'])

    print("Scores: \n", scores)


if __name__ == '__main__':
    inpath = Path("train_val_test_sets")
    output_basepath = Path("brr_results")
    outpaths = [output_basepath]
    outpaths = [output_basepath / str(i) for i in range(5)]

    if os.path.exists(output_basepath):
        print(f"Output directory {output_basepath} already exists. Deleting directory...")
        rmtree(output_basepath)

    if not os.path.exists(output_basepath):
        output_basepath.mkdir(parents=True, exist_ok=True)

    for outpath in outpaths:
        if not os.path.exists(outpath):
            outpath.mkdir(parents=True, exist_ok=True)

        # save command line printouts to file
        stdoutOrigin = sys.stdout
        sys.stdout = open(outpath / "log.txt", "w")
        main(inpath, outpath)

        sys.stdout.close()
        sys.stdout = stdoutOrigin
        print(open(outpath / "log.txt", "r").read())

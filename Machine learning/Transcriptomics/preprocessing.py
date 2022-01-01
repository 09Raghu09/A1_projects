"""
Discard genes with expression level 0 in more than 90% of cases
take in number of classes, default 2 = healthy/unhealthy
nice to have: also more classes and optional argument with column name identifiers
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self, inputpath, group_identifiers):
        self.data = pd.read_csv(inputpath, index_col=0, delimiter="\t")
        self.group_identifiers = group_identifiers
        self.nr_groups = len(self.group_identifiers)

        self.group_counts = [0 for group in range(self.nr_groups)]

        assert len(group_identifiers) == self.nr_groups

    def get_pandas_df(self):
        return self.data

    def count_groupsizes(self):
        for group_index, group_id_list in enumerate(self.group_identifiers):
            group_counter = 0
            for group_substr in group_id_list:
                subst_counter = [True for col in self.data.columns if group_substr.lower() in col.lower()]
                group_counter += len(subst_counter)
            if group_index == 0 and self.nr_groups == 2:
                self.group_counts[0] = group_counter
                self.group_counts[1] = len(self.data.columns) - group_counter
                return

            else:
                self.group_counts[group_index] = group_counter
        print(f"Group sizes: {self.group_counts}")

    def get_toy_example(self):
        export_cols_healthy = [col for col in self.data.columns if any([True for subst in self.group_identifiers[0] if subst.lower() in col.lower()])]
        export_cols_cancer = [col for col in self.data.columns if not any([True for subst in self.group_identifiers[0] if subst.lower() in col.lower()])]

        export_sample_healthy = random.sample(export_cols_healthy, 8)
        export_sample_cancer = random.sample(export_cols_cancer, 10)
        export_sample = export_sample_healthy + export_sample_cancer

        return self.data[export_sample][:150]

    def discard_expression_level_0_genes(self, threshold_amount_percent=0.9):
        print(f"Discarding genes from dataset if relative amount of expression level 0 is over 90%.")
        print(f"Dataframe shape before sorting out low expression rate genes: {self.data.shape}")
        low_qual_indices = [index for index, row in enumerate(self.data.values) if
                            len(np.where(row == 0)[0]) / self.data.columns.size >= threshold_amount_percent]
        self.data.drop(self.data.index[low_qual_indices], inplace=True)
        print(f"Dataframe shape after sorting out low expression rate genes: {self.data.shape}")

    def equalize_group_size(self, strategy="delete"):
        def col_is_in_group(col, group_index):
            if group_index == 1 and not any([True for subst in self.group_identifiers[0] if subst.lower() in col.lower()]):
                return True
            elif group_index == 0 and any(
                    [True for subst in self.group_identifiers[0] if subst.lower() in col.lower()]):
                return True
            else:
                return any([True for substr in self.group_identifiers[group_index] if substr.lower() in col.lower()])

        print(f"Dataframe shape before evening out groupsizes: {self.data.shape}")

        self.count_groupsizes()
        try:
            assert np.sum(self.group_counts) == len(self.data.columns)
        except AssertionError:
            raise AssertionError(
                f"Columns can not be seperated into given number of {self.nr_groups} groups. Either update group "
                f"identifiers or nr_groups variable.")

        print(f"Group counts: {self.group_counts}")
        bigger_group = [index for index, val in enumerate(self.group_counts) if val == max(self.group_counts)]
        if len(bigger_group) > 1:
            return
        else:
            bigger_group = bigger_group[0]

        if strategy == "delete":
            amount_to_delete = max(self.group_counts) - min(self.group_counts)
            column_list = [col for col in self.data.columns if col_is_in_group(col, bigger_group)]
            to_delete_colnames = random.sample(column_list, amount_to_delete)
            print(f"Randomly deleting {amount_to_delete} columns from group {bigger_group + 1}.")
            self.data.drop(to_delete_colnames, inplace=True, axis=1)
            self.count_groupsizes()
            print(f"Group sizes: {self.group_counts}")

        elif strategy == "bootstrap":
            difference = max(self.group_counts) - min(self.group_counts)
            amount_to_delete = int(difference / 2)
            amount_to_resample = difference - amount_to_delete
            column_list_delete = [col for col in self.data.columns if col_is_in_group(col, bigger_group)]
            column_list_resample = [col for col in self.data.columns if not col_is_in_group(col, bigger_group)]
            to_delete_colnames = random.sample(column_list_delete, amount_to_delete)
            to_bootstrap_colnames = random.choices(column_list_resample, k=amount_to_resample)

            # delete some
            print(f"Deleting {amount_to_delete} columns from group {bigger_group + 1}.")
            self.data.drop(to_delete_colnames, inplace=True, axis=1)
            # add some
            print(f"Resampling {amount_to_resample} columns from and to other group.")
            for colname in to_bootstrap_colnames:
                original = colname
                if colname in self.data.columns:
                    colname += "_c"
                while colname in self.data.columns:
                    colname += "c"
                self.data[colname] = self.data[original]
            self.count_groupsizes()

            # verify no group is more than one member bigger as the other groups
            assert not any([True for index, count in enumerate(sorted(self.group_counts)[1:]) if abs(self.group_counts[index] - count) > 1])
        else:
            raise LookupError(f"Strategy {strategy} not found in options.")

        self.count_groupsizes()
        print(self.group_counts)

        # make sure groups are as equal in size as possible
        print(f"Dataframe shape after evening out groupsizes: {self.data.shape}")


def main(in_path, out_path):
    group_identifiers = [["HD", "control"],
                         ["KRAS", "Unkown", "Liver", "Chol", "HBC", "Breast", "CRC", "Lung",
                          "Pancr", "NSCLC", "BrCa"]]

    preprocessor = Preprocessor(in_path, group_identifiers)

    preprocessor.discard_expression_level_0_genes()
    # preprocessor.equalize_group_size()
    preprocessor.equalize_group_size(strategy="bootstrap")

    print(f"Saving preprocessed data to {out_path}.")
    preprocessor.data.to_csv(out_path, sep="\t")


if __name__ == '__main__':
    data_dirpath = Path("../data")
    input_dir = data_dirpath / "input"
    output_dir = data_dirpath / "output"

    inputfilename = "GSE68086_TEP_data_matrix.txt"

    inputfilepath = input_dir / inputfilename
    outputfilename = Path(f"{inputfilepath.stem}_preprocessed.txt")
    if len(sys.argv) > 1:
        outputfilename = f"{outputfilename.stem}_{sys.argv[1]}.txt"
    outputfilepath = output_dir / outputfilename

    if os.path.exists(outputfilepath):
        print(f"Output path {outputfilepath} already exists. Deleting file...")
        os.remove(outputfilepath)

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    main(inputfilepath, outputfilepath)

import pymrmr
import pandas as pd

csv_file = "feature_list.csv"
features_df = pd.read_csv(csv_file,  delimiter="\t")
print(features_df.head(n=18))
pymrmr.mRMR(features_df, 'MIQ', 18)

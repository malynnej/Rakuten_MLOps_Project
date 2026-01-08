import pandas as pd
import os

import yaml
from box import ConfigBox

from sklearn.model_selection import train_test_split


class initialization:
    def __init__(self, config_path = "./src/features/config.yaml", params_path = "./src/features/params.yaml", dev_mode=False, test_obs=500):
        self.config_path = config_path
        self.params_path = params_path
        self.dev_mode = dev_mode
        self.test_obs = test_obs

    def read_yaml(self, path):
        with open(path) as yaml_file:
            content = yaml.safe_load(yaml_file)

            return ConfigBox(content)

    def load_data(self, params, paths):

        X = pd.read_csv(paths.Paths.X_train, index_col=0)
        y = pd.read_csv(paths.Paths.y_train, index_col=0)

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=params.Data.test_size,
                                                                random_state=params.Data.random_state,
                                                                stratify=y)

        df_train = X_train.join(y_train)
        df_test = X_test.join(y_test)

        if self.dev_mode:
            df_train = df_train.head(self.test_obs)
            df_test = df_test.head(self.test_obs)


        return df_train, df_test
    
    def create_path_columns(self, dataframe,paths):

        df = dataframe.copy()
        df["image_name"] = "image_" + df["imageid"].astype(str) + "_product_" + df["productid"].astype(str) + ".jpg"
        df["image_path"] = paths.Paths.image_train_path + "image_" + df["imageid"].astype(str) + "_product_" + df["productid"].astype(str) + ".jpg"
        df["image_path_preprocessed"] = paths.Paths.image_train_path_output + df["prdtypecode"].astype(str) + "/" + "image_" + df["imageid"].astype(str) + "_product_" + df["productid"].astype(str) + ".jpg"
        #df["image_path_greyscale"] = paths.Paths.image_train_path_greyscale + df["prdtypecode"].astype(str) + "/" + "image_" + df["imageid"].astype(str) + "_product_" + df["productid"].astype(str) + ".jpg"

        return df
    
    def combine_text_columns(self, dataframe):

        # Combine description and designation column
        # combined text column names - text
        # Avoiding repeating text occurs in both col
        # After combinining not dropping desig and description just for LLM
       
        df = dataframe.copy()

        df["description"] = df["description"].fillna(" ")
        indices_eq = df.index[df["description"] == df["designation"]]
        df.loc[indices_eq, "description"] = " "

        for i in df.index:
            if i not in indices_eq:
                df.at[i, "text"] = df.at[i, 'designation'] + "; "+ df.at[i, 'description']
            else: 
                df.at[i, "text"] = df.at[i, 'description']

        return df

    def initialize(self):
        paths = self.read_yaml(self.config_path)
        params = self.read_yaml(self.params_path)

        print("Execute following preprocessing steps:")
        if any(params.ExecFlags.values()):
            for key in params.ExecFlags:
                if params.ExecFlags[key]:
                    print(f">>> {key}")
        else:
            print(">>> None")

        print("Load data.")
        df_train, df_test = self.load_data(params, paths)

        print("Create image path columns.")
        df_train = self.create_path_columns(df_train, paths)
        df_test = self.create_path_columns(df_test, paths)

        print("Combine text columns.")
        df_train = self.combine_text_columns(df_train)
        df_test = self.combine_text_columns(df_test)

        return df_train, df_test, paths, params
    
    def save_files(self, dataframe, paths, mode, X_columns):

        df = dataframe.copy()

        # Save preprocessed data frame
        ## Check if save path exists. Create path if not.
        if not os.path.exists(paths.Paths.TextDataFrameSavePath) or not os.path.isdir(paths.Paths.TextDataFrameSavePath):
            os.makedirs(paths.Paths.TextDataFrameSavePath)
            print(f"created path: {paths.Paths.TextDataFrameSavePath}")

        X = df[X_columns]
        y = df["prdtypecode"]
        images = df[["image_name", "image_path", "image_path_preprocessed"]]

        X.to_csv(paths.Paths.TextDataFrameSavePath + "X_" + mode + ".csv")
        y.to_csv(paths.Paths.TextDataFrameSavePath + "y_" + mode + ".csv")
        images.to_csv(paths.Paths.TextDataFrameSavePath + "images_" + mode + ".csv")






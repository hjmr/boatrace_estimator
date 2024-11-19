import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import lightgbm as lgb
import shap

from const import DATA_LIST


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()
    return args


arg = parse_args()
data = pd.read_csv(arg.input, index_col=None, header=0, encoding="cp932")

i = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
data["rank_1"] = data["rank_1"].map(i)
data["rank_2"] = data["rank_2"].map(i)
data["rank_3"] = data["rank_3"].map(i)
data["rank_4"] = data["rank_4"].map(i)
data["rank_5"] = data["rank_5"].map(i)
data["rank_6"] = data["rank_6"].map(i)

X = data.drop(["rank_1", "rank_2", "rank_3", "rank_4", "rank_5", "rank_6"], axis=1)
T = data[["rank_1", "rank_2", "rank_3", "rank_4", "rank_5", "rank_6"]]

X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.3)

x_train = X_train[DATA_LIST].to_numpy()
t_train = T_train["rank_1"].to_numpy().astype(int)

x_test = X_test[DATA_LIST].to_numpy()
t_test = T_test["rank_1"].to_numpy().astype(int)

d_train = lgb.Dataset(x_train, label=t_train)
d_test = lgb.Dataset(x_test, label=t_test)

# specify your configurations as a dict
# fmt: off
params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "metric": "multi_error",
    "num_class": 6,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}
# fmt: on

# create model
model = lgb.LGBMClassifier(**params)  # scikit API

# train
print("Starting training...")
model.fit(x_train, t_train)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test, t_test)
explain = shap.Explanation(
    shap_values,
    base_values=explainer.expected_value,
    data=x_test,
    feature_names=DATA_LIST,
    output_names=["rank_1", "rank_2", "rank_3", "rank_4", "rank_5", "rank_6"],
)

for i in range(6):
    shap.plots.bar(explain[:, :, i], show=False)
    plt.title("Bar " + str(i))
    plt.show()

for i in range(6):
    shap.plots.beeswarm(explain[:, :, i], show=False)
    plt.title("Summary_Plot " + str(i))
    plt.show()

for i in range(6):
    shap.plots.decision(
        base_value=explainer.expected_value[i],
        shap_values=shap_values[:, :, i],
        feature_names=DATA_LIST,
        link="logit",
        show=False,
    )
    plt.title("Decision " + str(i))
    plt.show()

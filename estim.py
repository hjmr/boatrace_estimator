import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

x_train = X_train[DATA_LIST]
t_train = T_train["rank_1"]

x_test = X_test[DATA_LIST]
t_test = T_test["rank_1"]

dtrain = lgb.Dataset(x_train, t_train)
dtest = lgb.Dataset(x_test, t_test)

# fmt: off
params = {
    "objective": "multiclass",
    "metric"   : "multi_logloss",
    "num_class": 6,
    "random_state": 100
}
# fmt: on

model_1 = lgb.train(params, dtrain, valid_sets=[dtrain, dtest], callbacks=[lgb.early_stopping(stopping_rounds=10)])

explainer = shap.TreeExplainer(model_1, data=x_test)

shap_values = explainer.shap_values(x_test, approximate=True)
shap.summary_plot(shap_values=shap_values, features=x_test, plot_type="bar")

print(t_test.iloc[0, :])
print("")
# for i in range(6):
#    print("Class ", i)
#    shap.force_plot(explainer.expected_value[i], shap_values[i][0, :], x_test.iloc[0, :])

shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values, features=x_test)

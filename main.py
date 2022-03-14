from tokenize import Decnumber
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self) -> None:
        self.dataframe = pd.read_excel(
            "Ouse93-96 - Student.xlsx", sheet_name="Clean Data"
        )
        self.column_names = []
        self.dates = self.dataframe["Dates"]

    def get_data(self):
        return self.dataframe

    def get_number_of_columns(self):
        return len(self.dataframe.columns)

    def get_column_names(self):
        for i in range(1, self.get_number_of_columns()):
            self.column_names.append(self.dataframe.columns[i])

        return self.column_names

    def get_dates(self):
        return self.dates

    def create_predictors(self):
        predictors = [[] for _ in self.get_column_names()]
        count = 0
        for col in predictors:
            col.append(self.dataframe[self.column_names[count]])
            count += 1

        return predictors

    def plot_graph(self, predictors):
        plt.scatter(predictors[3][0], predictors[0][0], label="Crakehill x Skelton")
        # plt.scatter(predictors[3][0], predictors[1][0], label="Skip Bridge x Skelton")
        # plt.scatter(predictors[3][0], predictors[2][0], label="Westwick x Skelton")
        x = predictors[3][0]
        y = predictors[0][0]
        plt.title("Daily Flow")
        plt.ylabel("Predictor")
        plt.xlabel("Skelton")
        plt.legend()
        plt.show()


class MLP:
    def __init__(self, data) -> None:
        self.predictors = data.create_predictors()

    def pred(self):
        return self.predictors


data = Data()
mlp = MLP(data)
print(mlp.pred())

# predictors = data.create_predictors()

# data.plot_graph(predictors)  # plot graph

# training data: [:732]
# validation data: [732:976]
# testing data: [976:]

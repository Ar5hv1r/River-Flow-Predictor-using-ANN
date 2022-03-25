import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


class Data:
    def __init__(self) -> None:
        self.dataframe = pd.read_excel(
            "Ouse93-96 - Student.xlsx", sheet_name="Clean Data"
        )
        self.column_names = []
        self.dates = self.dataframe["Dates"]

    def get_predictand(self):
        predictand = list(self.dataframe["Skelton"].head(730).values)
        return predictand

    def get_predictand_length(self, predictand):
        return len(predictand)

    def get_columns(self):
        return list(self.dataframe.columns.values)

    def get_data(self):
        return self.dataframe

    def get_number_of_columns(self):
        return len(self.dataframe.columns)

    def get_column_names(self):
        self.column_names = list(self.dataframe.columns.values)
        return self.column_names

    def get_dates(self):
        return self.dates

    def get_number_of_values(self, predictors):
        return len(predictors[0])

    def get_predictors(self):
        predictors = []
        for i in range(1, self.get_number_of_columns() - 1):
            predictors.append(
                (
                    self.dataframe[self.get_column_names()[i]].head(730).values
                )  # training data
            )

        return predictors

    def standardise(self, predictors):
        copy = predictors[:]
        for i in range(len(predictors)):
            for j in range(len(predictors[i])):
                predictors[i][j] = (
                    0.8 * ((copy[i][j] - min(copy[i])) / (max(copy[i]) - min(copy[i])))
                    + 0.1
                )

        return predictors

    def standardise_predictand(self, predictand):
        copy = predictand[:]
        for i in range(len(predictand)):
            predictand[i] = (
                0.8 * ((copy[i] - min(copy)) / (max(copy) - min(copy))) + 0.1
            )

        return predictand


class Layer:
    def __init__(self, layer_sizes) -> None:
        self.weight_shapes = [
            elem for elem in zip(layer_sizes[1:], layer_sizes[:-1])
        ]  # list holding weight sizes of each layer
        self.weights = [
            np.random.uniform(-2 / s[-1], 2 / s[-1], s) for s in self.weight_shapes
        ]
        self.biases = [np.zeros((s)) for s in layer_sizes[1:]]
        self.learning_rate = 0.001
        self.rmse_numerator = 0
        self.rmse_values = []

    def predict(self, inputs, predictand):  # inputs to netwrok initially
        self.node_values = []
        for weights, biases in zip(self.weights, self.biases):
            inputs = self.forward_sigmoid(np.matmul(weights, inputs) + biases)
            self.node_values.append(inputs)
        self.rmse_numerator += (self.node_values[-1] - predictand) ** 2
        return inputs

    @staticmethod
    def forward_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def backward_sigmoid(self, x):
        return x * (1 - x)

    def backpropagation(self, predictand):
        self.output_delta = (predictand - self.node_values[-1]) * (
            self.backward_sigmoid(self.node_values[-1])
        )
        self.hidden_deltas = [self.output_delta]
        self.layer_deltas = []
        for i in range(len(self.weights) - 2, -1, -1):
            self.layer_deltas.clear()
            for j in range(len(self.weights[i])):
                delta_j = sum(
                    self.weights[i][j] * self.node_values[i][j]
                ) * self.backward_sigmoid(self.node_values[i][j])

                self.layer_deltas.append(delta_j)
            self.hidden_deltas.append(self.layer_deltas)
        self.hidden_deltas = self.hidden_deltas[::-1]
        # print(f"Weights Before: {self.weights}")

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] += (
                    self.learning_rate
                    * float(self.hidden_deltas[i][j])
                    * float(self.node_values[i][j])
                )

        # print(f"Weights After: {self.weights}")

        # print(f"Bias Before: {self.biases}")

        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j] -= self.learning_rate * self.hidden_deltas[i][j]

        # print(f"Bias After: {self.biases}")

    def RMSE(self, n):
        rmse = (self.rmse_numerator / n) ** 0.5
        print(rmse)
        self.rmse_values.append(rmse)

    def getRMSE(self):
        return self.rmse_values


if __name__ == "__main__":
    start_time = time.time()
    layer_sizes = (3, 5, 5, 1)
    data = Data()
    predictors = data.get_predictors()
    standardised = data.standardise(predictors)
    layer = Layer(layer_sizes)

    predictand = data.get_predictand()
    predictand = data.standardise_predictand(predictand)

    for i in range(500):
        layer.rmse_numerator = 0
        for j in range(len(standardised[0])):
            inputs = [val[j] for val in standardised]
            layer.predict(inputs, predictand[j])
            layer.backpropagation(predictand[j])

        layer.RMSE(data.get_predictand_length(predictand))
    # print(layer.getRMSE())
    print("--- %s seconds ---" % (time.time() - start_time))
    x = list(range(1, 501))
    plt.xlabel("Epochs")
    plt.ylabel("RMSE Error")
    plt.plot(x, layer.getRMSE())
    plt.show()

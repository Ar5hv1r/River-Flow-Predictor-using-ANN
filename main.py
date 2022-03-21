import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


class Data:
    def __init__(self) -> None:
        self.dataframe = pd.read_excel(
            "Ouse93-96 - Student.xlsx", sheet_name="Clean Data"
        )
        self.column_names = []
        self.dates = self.dataframe["Dates"]

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

    def plot_graph(self, predictors):
        # plt.scatter(predictors[3][0], predictors[0][0], label="Crakehill x Skelton")
        # plt.scatter(predictors[3][0], predictors[1][0], label="Skip Bridge x Skelton")
        plt.scatter(predictors[0], predictors[3], label="Westwick x Skelton")
        # x = predictors[4]
        # y = predictors[3]
        plt.title("Daily Flow")
        plt.ylabel("Predictor")
        plt.xlabel("Skelton")
        plt.legend()

        # theta = np.polyfit(x, y, 1)  # linear line of best fit
        # y_line = theta[1] + theta[0] * x
        # plt.plot(x, y_line, "r")

        plt.show()

    # 0.8*(rawval - min / max-min) + 0.1
    def standardise(self, predictors):
        for i in range(len(predictors)):
            for j in range(len(predictors[i])):
                predictors[i][j] = 0.8 * (
                    (predictors[i][j] - min(predictors[i]))
                    / (max(predictors[i] - min(predictors[i])) + 0.1)
                )

        return predictors


class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.weights = 0.01 * np.random.randn(number_of_inputs, number_of_neurons)
        self.biases = np.zeros((1, number_of_neurons))

    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Sigmoid:
    def __init__(self) -> None:
        pass

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output


class ReLU:
    def __init__(self) -> None:
        pass

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output


class Softmax:
    def __init__(self) -> None:
        pass

    def forward(self, inputs):
        e_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalised = e_vals / np.sum(e_vals, axis=1, keepdims=True)
        self.output = normalised


class LMSE:
    def __init__(self) -> None:
        pass

    def forward(self, predicted_output, actual_output):
        self.output = (predicted_output - actual_output) ** 2

        return self.output


data = Data()
predictors = data.get_predictors()
standardised = data.standardise(predictors)
print(standardised)

"""
layer1 = Layer(data.get_number_of_values(predictors), 5)
sig1 = Sigmoid()
layer1.forward_pass(predictors)
layer1sig = sig1.forward(layer1.output)

sig2 = Sigmoid()
layer2 = Layer(5, 5)
layer2.forward_pass(layer1sig)
layer2sig = sig2.forward(layer2.output)

sig3 = Sigmoid()
outputLayer = Layer(5, 1)
outputLayer.forward_pass(layer2sig)
outputSig = sig3.forward(outputLayer.output)

loss = LMSE()
loss.forward(outputSig, predictors)
print(loss.output)
"""

# relu1 = ReLU()
# layer2 = Layer(3, 3)
# relu2 = Softmax()
# layer1.forward_pass(x)
# relu1.forward(layer1.output)
# layer2.forward_pass(relu1.output)
# relu2.forward(layer2.output)
# print(relu2.output[:5])
# softmax = Softmax()
# softmax.forward([[1, 2, 3]])
# print(softmax.output)
# print(sum(sum(softmax.output)))
# print(sum(softmax.output))


# create data object
# data = Data()
# predictors = data.get_predictors()
# layer1 = Layer(data.get_number_of_values(predictors), 4)
# relu = ReLU()
# layer1.forward_pass(predictors)
# print(relu.forward(layer1.output))

# layer2 = Layer(4, 4)
# layer2.forward_pass(layer1.output)
# print(layer1.output)
# print(layer2.output)


# layer2 = Layer(3, 5)
# layer2.forward_pass(layer1.output)
# print(layer2.output)

# predictors = data.get_predictors()

# data.plot_graph(predictors)

# predictors = data.create_predictors()

# data.plot_graph(predictors)  # plot graph

# training data: [:732]
# validation data: [732:976]
# testing data: [976:]

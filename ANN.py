# necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data:
    def __init__(self) -> None:
        """Initialize basic values"""
        self.dataframe = pd.read_excel(
            "Ouse93-96 - Student.xlsx", sheet_name="Clean Data"
        )
        self.column_names = []
        self.dates = self.dataframe["Dates"]

    def get_predictand(self):
        """Returns training data values that are within the predictand
        'Skelton' can be altered to the column name for your predictand

        Returns:
            list: list of n values representing training data
        """
        predictand = list(self.dataframe["Skelton"].head(730).values)
        return predictand

    def get_predictand_length(self, predictand):
        """Returns length of a predictand

        Args:
            predictand (list): Holds values of precictand

        Returns:
            int: Integer value holding length of a predictand
        """
        return len(predictand)

    def get_columns(self):
        """Returns column values from inputted Excel sheet

        Returns:
            list: returns 2D array of values from all columns
        """
        return list(self.dataframe.columns.values)

    def get_data(self):
        """Retrieves all data from Excel sheet

        Returns:
            dataframe: returns a Dataframe object containing all values and columns from Excel sheet
        """
        return self.dataframe

    def get_number_of_columns(self):
        """Retrieves number of columns in Excel sheet

        Returns:
            int: Integer value holding number of columns in Excel sheet
        """
        return len(self.dataframe.columns)

    def get_column_names(self):
        """Retrieves names of columns in Excel sheet

        Returns:
            list: List holding string values of column names in Excel sheet
        """
        self.column_names = list(self.dataframe.columns.values)
        return self.column_names

    def get_dates(self):
        """Returns dates in Excel sheet

        Returns:
            Series: Returns a Series object holding all dates in the Excel sheet
        """
        return self.dates

    def get_number_of_values(self, predictors):
        """Returns number of values within a predictor

        Args:
            predictors (Multi-Dimensional Array): Holds column names and values associated with it

        Returns:
            int: Integer value holding number of values within a column
        """
        return len(predictors[0])

    def get_predictors(self):
        """Returns values of all predictors

        Returns:
            list: Multi-dimensional Array that holds values of each column in a separate array
        """
        predictors = []
        for i in range(1, self.get_number_of_columns() - 1):
            predictors.append(
                (
                    self.dataframe[self.get_column_names()[i]].head(730).values
                )  # training data
            )
        # print(f"Predictors: {predictors}")
        return predictors

    def get_validation_data(self):
        validation_data = []
        for i in range(1, self.get_number_of_columns() - 1):
            validation_data.append(
                self.dataframe[self.get_column_names()[i]][731:1127].values
            )
        # print(validation_data)
        return validation_data

    def get_validation_predictand(self):
        validation_predictand = list(self.dataframe["Skelton"][731:1127].values)
        return validation_predictand

    def standardise(self, predictors):
        """Standardises data between a certain range

        Args:
            predictors (list): Input list of predictors

        Returns:
            list: Returns list of standardised values, typically between 0 and 1
        """
        copy = predictors[:]
        for i in range(len(predictors)):
            for j in range(len(predictors[i])):
                predictors[i][j] = (
                    0.8 * ((copy[i][j] - min(copy[i])) / (max(copy[i]) - min(copy[i])))
                    + 0.1
                )

        return predictors

    def standardise_predictand(self, predictand):
        """Standardises values in the predictand

        Args:
            predictand (list): Holds all values in predictand

        Returns:
            list: Returns list of standardised values, typically between 0 and 1
        """
        copy = predictand[:]
        for i in range(len(predictand)):
            predictand[i] = (
                0.8 * ((copy[i] - min(copy)) / (max(copy) - min(copy))) + 0.1
            )

        return predictand

    def standardise_validation(self, validation_data):
        copy = validation_data[:]
        for i in range(len(validation_data)):
            for j in range(len(validation_data[i])):
                validation_data[i][j] = (
                    0.8 * ((copy[i][j] - min(copy[i])) / (max(copy[i]) - min(copy[i])))
                    + 0.1
                )

        return validation_data

    def standardise_predictand_validation(self, validation_predictand):
        copy = validation_predictand[:]
        for i in range(len(validation_predictand)):
            validation_predictand[i] = (
                0.8 * ((copy[i] - min(copy)) / (max(copy) - min(copy))) + 0.1
            )

        return validation_predictand


class Layer:
    def __init__(self, layer_sizes) -> None:
        """Initialize basic values for class

        Args:
            layer_sizes (tuple): Tuple holding integer values that represent number of neurons in each layer
        """
        self.weight_shapes = [
            elem for elem in zip(layer_sizes[1:], layer_sizes[:-1])
        ]  # generates a list of tuples holding the matrix size for each weight
        self.weights = [
            np.random.uniform(-2 / s[-1], 2 / s[-1], s) for s in self.weight_shapes
        ]  # randomises weights and generates a matrix based on matrix sizes
        self.biases = [
            np.zeros((s)) for s in layer_sizes[1:]
        ]  # generates a zero matrix to represent biases
        self.learning_rate = 0.001  # sets learning rate for MLP
        self.rmse_numerator = 0
        self.rmse_values = []
        self.validate_numerator = 0
        self.validate_RMSE_values = []

    def validate(self, validation_data, validation_predictand):
        for weights, biases in zip(self.weights, self.biases):
            validation_data = self.forward_sigmoid(
                np.matmul(weights, validation_data) + biases
            )

        self.validate_numerator += (validation_data - validation_predictand) ** 2

    def predict(self, inputs, predictand):
        """Forward Pass through Neural Network (NN)

        Args:
            inputs (list): Standardised values for input layer
            predictand (list): Standardised values for the 'actual output'

        Returns:
            list: Returns values at each node after weights and biases have been applied alongside an activation function
        """
        self.node_values = []
        for weights, biases in zip(self.weights, self.biases):
            inputs = self.forward_sigmoid(np.matmul(weights, inputs) + biases)
            self.node_values.append(inputs)
        self.rmse_numerator += (self.node_values[-1] - predictand) ** 2
        return inputs

    @staticmethod
    def forward_sigmoid(x):
        """Sigmoid Function - Forward Pass

        Args:
            x (list): list of inputs into the network

        Returns:
            float: value between 0 and 1 denoting how likely a neuron is to be turned 'on' or 'off'
        """
        return 1 / (1 + np.exp(-x))

    def backward_sigmoid(self, x):
        """Backward Pass Sigmoid - Calculates derivative of Sigmoid function

        Args:
            x (list): list of inputs that have had the activation function applied to them

        Returns:
            float: derivative of Sigmoid function
        """
        return x * (1 - x)

    def backpropagation(self, predictand):
        """Backpropagation algorithm that backtracks through the network, updating weights and biases accordingly

        Args:
            predictand (list): Predictand that the network should desire to reach
        """
        self.output_delta = (predictand - self.node_values[-1]) * (
            self.backward_sigmoid(self.node_values[-1])
        )  # calculate delta value at the output node using derivative of sigmoid function
        self.hidden_deltas = []
        self.hidden_deltas.append(
            self.output_delta
        )  # add output delta to final delta list
        self.layer_deltas = []

        # loop through weights for the network backwards, not including weights going from the final hidden layer to the output layer
        for i in range(len(self.weights) - 2, -1, -1):
            del self.layer_deltas[:]  # clear weights for the current layer
            for j in range(len(self.weights[i])):
                self.layer_deltas.append(self.weights[i][j] * self.node_values[i][j])
            self.layer_deltas = list(
                map(sum, self.layer_deltas)
            )  # sum weights of nodes and the product of the value at the node
            self.layer_deltas = list(
                map(
                    (self.backward_sigmoid(self.node_values[i][j])).__mul__,
                    self.layer_deltas,
                )
            )  # apply derivative and multiple result
            self.hidden_deltas.append(
                self.layer_deltas[:]
            )  # append a copy of the list to the final delta list
        self.hidden_deltas = self.hidden_deltas[
            ::-1
        ]  # reverse list for easier calculations

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] += (
                    self.learning_rate
                    * float(self.hidden_deltas[i][j])
                    * float(self.node_values[i][j])
                )  # update all weights

        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j] += (
                    self.learning_rate * self.hidden_deltas[i][j]
                )  # update all biases

    def RMSE(self, n):
        """Calculates RMSE (Root Squared Mean Error) for each epoch

        Args:
            n (int): number of values passed into the function
        """
        rmse = (self.rmse_numerator / n) ** 0.5
        self.rmse_values.append(rmse)

    def validateRMSE(self, n):
        rmse = (self.validate_numerator / n) ** 0.5
        print(rmse)
        self.validate_RMSE_values.append(rmse)
        return rmse

    def getRMSE(self):
        """Retrieves value of RMSE calculation

        Returns:
            list: list containing floats of the result of the RMSE calculations
        """
        return self.rmse_values


if __name__ == "__main__":
    rmse_results = []
    # adjustable layer sizes --> first and last numbers are input and output layers respectively
    # middle numbers are hidden nodes
    layer_sizes = (3, 5, 4, 7, 1)

    data = Data()

    # get predictors and standardise
    predictors = data.get_predictors()
    standardised = data.standardise(predictors)

    layer = Layer(layer_sizes)

    # get predictand and standardise
    predictand = data.get_predictand()
    predictand = data.standardise_predictand(predictand)

    # get validation data and standardise
    validation_data = data.get_validation_data()
    validation_data = data.standardise_validation(validation_data)

    validation_predictand = data.get_validation_predictand()
    validation_predictand = data.standardise_predictand_validation(
        validation_predictand
    )
    flag = False
    # loop through n epochs
    for i in range(1000):
        # reset RMSE numerator after each epoch
        layer.rmse_numerator = 0
        print(f"Epoch: {i}")
        for j in range(len(standardised[0])):
            # create and pass in necessary values
            inputs = [val[j] for val in standardised]
            layer.predict(inputs, predictand[j])
            layer.backpropagation(predictand[j])
        if not (i % 50):
            layer.validate_numerator = 0
            for k in range(len(validation_data[0])):
                validation_input = [val[k] for val in validation_data]
                layer.validate(validation_input, validation_predictand[k])
            rmse_results.append(
                layer.validateRMSE(data.get_predictand_length(validation_predictand))
            )

            if len(rmse_results) > 1:
                print(rmse_results)
                if rmse_results[-1] > rmse_results[-2]:
                    flag = True

        if flag:
            break

        layer.RMSE(data.get_predictand_length(predictand))
    # plot graph of RMSE against epochs
    x = list(range(0, i + 1))
    plt.xlabel("Epochs")
    plt.ylabel("RMSE Error")
    plt.plot(x, layer.getRMSE())
    plt.show()

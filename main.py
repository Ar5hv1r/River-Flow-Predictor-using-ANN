import pandas as pd
import matplotlib.pyplot as plt


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
        predictors = [[] for col in self.get_column_names()]
        count = 0
        for col in predictors:
            col.append(self.dataframe[self.column_names[count]])
            count += 1

        return predictors

    def plot_graph(self, predictors):
        plt.plot(self.dates, predictors[0][0], label="Crakehill")
        plt.plot(self.dates, predictors[1][0], label="Skip Bridge")
        plt.plot(self.dates, predictors[2][0], label="Westwick")

        plt.title("Daily Flow")
        plt.ylabel("Mean Daily Flow - Cumecs")
        plt.xlabel("Date")
        plt.legend()
        plt.show()


data = Data()
predictors = data.create_predictors()
print(data.get_dates())
data.plot_graph(predictors)


"""
dates = dataframe["Dates"]
predictor1 = dataframe["Crakehill"]  # all values in column
predictor1_training = dataframe["Crakehill"][:732]  # 2 years of training data
predictor1_validation = dataframe["Crakehill"][732:976]  # 1 year of validation data
predictor1_testing = dataframe["Crakehill"][976:]  # testing data

predictor2 = dataframe["Skip Bridge"]  # all values in column
predictor2_training = dataframe["Skip Bridge"][:732]  # 2 years of training data
predictor2_validation = dataframe["Skip Bridge"][732:976]  # 1 year of validation data
predictor2_testing = dataframe["Skip Bridge"][976:]  # testing data

predictor3 = dataframe["Westwick"]  # all values in column
predictor3_training = dataframe["Westwick"][:732]  # 2 years of training data
predictor3_validation = dataframe["Westwick"][732:976]  # 1 year of validation data
predictor3_testing = dataframe["Westwick"][976:]  # testing data

plt.plot(dates, predictor1, label="Crakehill")
plt.plot(dates, predictor2, label="Skip Bridge")
plt.plot(dates, predictor3, label="Westwick")

plt.title("Daily Flow")
plt.ylabel("Mean Daily Flow - Cumecs")
plt.xlabel("Date")
plt.legend()
plt.show()"""

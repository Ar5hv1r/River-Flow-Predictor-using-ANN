import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_excel("Ouse93-96 - Student.xlsx", sheet_name="Clean Data")
dates = dataframe["Dates"]
predictor1 = dataframe["Crakehill"]  # all values in column
predictor1_training = dataframe["Crakehill"][:732]  # 2 years of training data
predictor1_validation = dataframe["Crakehill"][732:976]  # 1 year of validation data
predictor1_testing = dataframe["Crakehill"][976:]  # testing data

plt.plot(dates, predictor1)
plt.ylabel("Mean Daily Flow - Cumecs")
plt.xlabel("Date")
plt.show()

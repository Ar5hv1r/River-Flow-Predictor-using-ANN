import pandas as pd
import pprint

printer = pprint.PrettyPrinter()

workbook = "Ouse93-96 - Student.xlsx"
sheet = pd.read_excel(workbook, sheet_name="1993-96")

crakehill = sheet["Mean Daily Flow - Cumecs"]
printer.pprint(crakehill)

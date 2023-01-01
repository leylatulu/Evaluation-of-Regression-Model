# required packages
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)

# Years of experience and salary information of the employees are given.
years_of_experience = np.array([5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1])
salary = np.array([600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380])

# assign bias and weight values
bias = 275
weight = 90

# define regression model
def reg_model(bias, weight):
    return bias + weight * years_of_experience

# find predicted salary
pred_salary = reg_model(bias, weight)

# calculate errors
df = pd.DataFrame({"Years of Experience (x)": years_of_experience,
                   "Salary (y)": salary,
                   "Salary Prediction (y')": pred_salary,
                   "Error (y-y')": salary - pred_salary,
                   "Squares of Error": (salary - pred_salary)**2,
                   "Absolute Error |y-y'|": np.abs(salary - pred_salary)})

# show results
eval_metris = pd.DataFrame({"Metrics": ["RMSE", "MSE", "MAE"],
                            "Results": [np.sqrt(np.mean(df["Squares of Error"])),
                                        np.mean(df["Squares of Error"]),
                                        np.mean(df["Absolute Error |y-y'|"])]})

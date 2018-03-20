import matplotlib.pyplot as plt
import numpy as np


ft_lauderdale_filename = "ft_lauderdale_beach.csv"

ft_lauderdale_file = open(ft_lauderdale_filename)

data = ft_lauderdale_file.read()
# print(len(data))
# print(data[:100])

ft_lauderdale_file.close()

lines = data.split('\n')
# print(len(lines))
# print(lines[0])

labels = lines[0]
values = lines[1:]
n_values = len(values)
# print(labels)
# for i_row in range(10):
#     print(values[i_row])

j_year = 1
j_month = 2
j_day = 3
j_max_temp = 5
year = []
month = []
day = []
max_temp = []
for i_row in range(n_values):
    split_values = values[i_row].split(',')
    # print(split_values)
    if len(split_values) >= j_max_temp:
        year.append(int(split_values[j_year]))
        month.append(int(split_values[j_month]))
        day.append(int(split_values[j_day]))
        max_temp.append(float(split_values[j_max_temp]))

# for i_day in range(100):
#     print(max_temp[i_day])

# plt.plot(max_temp)
# plt.show()

i_mid = len(max_temp) // 2
temps = np.array(max_temp[i_mid:])
temps[np.where(temps == -99.9)] = np.nan
# plt.plot(temps, color="black", marker='.', linestyle="none")
# plt.show()

# Find the first non-nan
# print(np.where(np.isnan(temps))[0])
# print(np.where(np.logical_not(np.isnan(temps)))[0][0])
i_start = np.where(np.logical_not(np.isnan(temps)))[0][0]
temps = temps[i_start:]
# print(np.where(np.isnan(temps))[0])
i_nans = np.where(np.isnan(temps))[0]
# print(np.diff(i_nans))
# np.where(np.logicalnot(np.isnan(temps)))[0])
for i in range(temps.size):
    if np.isnan(temps[i]):
        temps[i] = temps[i - 1]

# plt.plot(temps)
# plt.show()

# print(np.where(np.isnan(temps))[0])
# plt.plot(temps[:-1], temps[1:], color="black", marker='.', linestyle="none")
# plt.show()

def scatter(x, y):
    """
    Make a scatter plot with jitter.
    """
    x_jitter = x + np.random.normal(size=x.size, scale=.5)
    y_jitter = y + np.random.normal(size=y.size, scale=.5)
    plt.plot(
        x_jitter,
        y_jitter,
        color="black",
        marker='.',
        linestyle="none",
        alpha=.05,
    )
    plt.show()

shift = 3
# print(np.corrcoef(temps[:-shift], temps[shift:]))


autocorr = []
for shift in range(1, 9):
    correlation = np.corrcoef(temps[:-shift], temps[shift:])
    autocorr.append(correlation[1, 0])

# plt.plot(autocorr)
# plt.show()

model_coefficients = np.polyfit(temps[:-3], temps[3:], 1)
# print(model_coefficients)
actuals = temps[3:]
predictions = temps[:-3] * model_coefficients[0] + model_coefficients[1]
errors = actuals - predictions

# scatter(actuals, predictions)
# scatter(actuals, errors)

actuals = temps[4:]
predictors = temps[1:-3]
model_coefficients = np.polyfit(predictors, actuals, 1)
# print(model_coefficients)
predictions = predictors * model_coefficients[0] + model_coefficients[1]
errors = actuals - predictions
# scatter(actuals, errors)
print('mean average error', np.sum(np.abs(errors)) / errors.size)

day_4_temps = temps[:-4]
# scatter(day_4_temps, errors)
# scatter(- predictors + day_4_temps, errors)
# print(np.corrcoef(day_4_temps, errors))

def day_of_year(year, month, day):
    """
    Calculate the day of the year.
    January 1 = 0

    Parameters
    ----------
    year: int
    month: int
    day: int

    Returns
    -------
    day_of_year: int
    """
    days_per_month = np.array([
        31,  # January
        28,  # February
        31,  # March
        30,  # April
        31,  # May
        30,  # June
        31,  # July
        31,  # August
        30,  # September
        31,  # October
        30,  # November
        31,  # December
    ])
    # For leap years
    if year % 4 == 0:
        days_per_month[1] += 1
    # if month == 0:
    #     day_of_year = day
    # else:
    day_of_year = np.sum(np.array(days_per_month[:month-1])) + day - 1
    return day_of_year

def predict(year, month, day, three_day_temp):
    """
    Predict the temperature at Ft. Lauderdale Beach.

    Parameters
    ----------
    year: int
    month: int
        January = 1, etc.
    day: int
    three_day_temp: float
        The high temperature at Ft. Lauderdale Beach three days
        before the trip. 

    Returns
    -------
    predicted_temp: float 
    recommendation: boolean
        If True, then book your tickets.
    """

    i_day = day_of_year(year[i_row], month[i_row], day[i_row])
    median_temp = median_temp_calendar[i_day]
    prediction = model[0] + model[1] * three_day_temp + model[2] * median_temp
    return prediction, prediction > target_temp

d_o_y = np.zeros(temps.size)
for i_row in range(temps.size):
    d_o_y[i_row] = day_of_year(year[i_row], month[i_row], day[i_row])

# scatter(d_o_y, temps)

median_temp_calendar = np.zeros(366)
ten_day_medians = np.zeros(temps.size)
for i_day in range(0, 365):
    low_day = i_day - 5 
    high_day = i_day + 4
    if low_day < 0:
        low_day += 365
    if high_day > 365:
        high_day += -365
    if low_day < high_day:
        i_window_days = np.where(
            np.logical_and(d_o_y >= low_day, d_o_y <= high_day)) 
    else: 
        i_window_days = np.where(
            np.logical_or(d_o_y <= low_day, d_o_y >= high_day)) 

    ten_day_median = np.median(temps[i_window_days])
    median_temp_calendar[i_day] = ten_day_median
    ten_day_medians[np.where(d_o_y == i_day)] = ten_day_median
        # np.sum(temps[i_window_days]) / i_window_days[0].size
    if i_day == 364:
        ten_day_medians[np.where(d_o_y == 365)] = ten_day_median
        median_temp_calendar[365] = ten_day_median

ten_day_medians_predictor = ten_day_medians[4:]
# print(ten_day_medians.size, np.unique(ten_day_medians), ten_day_medians)
# scatter(ten_day_medians, temps)
# scatter(ten_day_medians[4:], errors)

# print(np.corrcoef(errors, ten_day_medians_predictor))
model_coefficients_2 = np.polyfit(ten_day_medians_predictor, errors, 1)
# print(model_coefficients_2)
predictions_2 = (ten_day_medians_predictor * model_coefficients_2[0] +
                 model_coefficients_2[1])
errors_2 = errors - predictions_2
# scatter(actuals, errors_2)
# print('mean average error', np.sum(np.abs(errors_2)) / errors_2.size)
total_predictions = predictions + predictions_2

model = [model_coefficients[0] + model_coefficients_2[0],
    model_coefficients[1],
    model_coefficients_2[1]]


# Compare predictions three days away vs. actuals for above/below 85.
sensitivity = []
targets = np.arange(84, 90)
for target in targets:
    i_warm = np.where(actuals > 85)[0]
    i_warm_predictions = np.where(total_predictions > target)[0]
    n_true_positives = np.intersect1d(i_warm, i_warm_predictions).size
    n_false_negatives = np.setdiff1d(i_warm, i_warm_predictions).size
    n_false_positives = np.setdiff1d(i_warm_predictions, i_warm).size
    n_true_negatives = (actuals.size - n_true_positives -
                        n_false_positives - n_false_negatives)
    # print("Accurately predicted warm", n_true_positives, "times")
    # print("Predicted cold when it was warm", n_false_negatives, "times")
    # print("Predicted warm when it was cold", n_false_positives, "times")
    # print("Accurately predicted cold", n_true_negatives, "times")

    sensitivity.append(
        n_true_positives / (n_true_positives + n_false_positives))
    # print("Fraction of warm trips", sensitivity[-1])

plt.plot(targets, sensitivity, marker='+')
plt.xlabel('Temperature target')
plt.ylabel('Fraction of warm trips')
plt.show()

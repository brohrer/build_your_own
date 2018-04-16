"""
Given a date (month, day and year) predict whether
the high temperature in Fort Lauderdale
is likely to be 85 degrees or higher.
This will determine whether you purchase plane tickets
for a trip.
"""
import sys

import predict_weather as pw


def decide(test_year=2015, test_month=7, test_day=25, test_temp=95):
    """
    Parameters
    ----------
    test_year, test_month, test_day: ints
    test_temp: float

    Returns
    -------
    decision: boolean
    """
    predictor = pw.Predictor()
    prediction = predictor.predict(
        test_year, test_month, test_day, test_temp) 
    print('For year=', test_year,
          ', month=', test_month,
          ', day=', test_day,
          ', temp=', test_temp)
    print('predicted temperature is', prediction) 

    if prediction >= 90:
        decision = True
    else:
        decision = False

    return decision


if __name__ == '__main__':

    if len(sys.argv) < 5:
        print('Try it like this:')
        print('    python buy_tickets.py <year> <month> <day> <temp>')
    else:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        day = int(sys.argv[3])
        temp = float(sys.argv[4])
        decision = decide(
            test_year=year, test_month=month, test_day=day, test_temp=temp)
        if decision == True:
            print("Buy those tickets.")
        else:
            print("Not this time.")

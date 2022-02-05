import numpy as np
import pandas as pd
from scipy.optimize import linprog


def get_all_shifts():
    """Create all available shifts.

    Returns:
        Dataframe with all available shifts.
    """
    all_shifts = []

    for i in range(4, 9):
        for j in range(11 - i + 1):
            temp_list = []
            temp_list.extend([0] * j)
            temp_list.extend([1] * i)
            last_zero_slice = 11 - j - i
            temp_list.extend([0] * last_zero_slice)
            all_shifts.append(temp_list)
    return np.array(all_shifts)


def get_best_shifts(courier_distribution, day, area):
    """Create all available shifts.

    Args:
        courier_distribution (DataFrame): couriers distribution by hours.
        day (Date): date to compute shifts.
        area (int): delivery area id.
    Returns:
        Dataframe with computed couriers shifts.
    """
    day = pd.to_datetime(day)
    all_shifts = get_all_shifts()
    courier_distribution = [-item for item in courier_distribution]
    hour_cost = [1]*30
    res = linprog(c=hour_cost, A_ub=-all_shifts.T, b_ub=courier_distribution, method='simplex')
    res = res.x.astype(int)
    start_hour = 10

    shifts = []
    for i in res.nonzero()[0]:
        for count in range(res[i]):
            shifts.append(all_shifts[i])

    n_shifts = list(range(1, len(shifts) + 1))
    shifts = (np.array(shifts).T * np.array(n_shifts)).T

    result = []
    for shift in shifts:
        non_zeros = shift.nonzero()[0]
        result.append({
            'delivery_area_id': area,
            'shift_start_date': day + pd.Timedelta(hours=start_hour + non_zeros[0]),
            'shift_end_date': day + pd.Timedelta(hours=start_hour + non_zeros[-1]),
        })
    return pd.DataFrame(result)

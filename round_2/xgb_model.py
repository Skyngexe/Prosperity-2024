import numpy as np

def xgb_tree(x, num_booster):
    if num_booster == 0:
        state = 0
        if state == 0:
            state = (1 if x['HUMIDITY']<88.3680573 else 2)
            if state == 1:
                state = (3 if x['HUMIDITY']<78.2046661 else 4)
                if state == 3:
                    state = (7 if x['SUNLIGHT']<3026.52051 else 8)
                    if state == 7:
                        return 3.62106371
                    if state == 8:
                        return 5.50033808
                if state == 4:
                    state = (9 if x['SUNLIGHT']<3266.39355 else 10)
                    if state == 9:
                        return 8.07950878
                    if state == 10:
                        return 7.04799986
            if state == 2:
                state = (5 if x['SUNLIGHT']<4113.55762 else 6)
                if state == 5:
                    state = (11 if x['SUNLIGHT']<3666.51733 else 12)
                    if state == 11:
                        return 8.39746952
                    if state == 12:
                        return 9.73084068
                if state == 6:
                    state = (13 if x['HUMIDITY']<94.0856781 else 14)
                    if state == 13:
                        return 12.1782608
                    if state == 14:
                        return 10.5517931
    elif num_booster == 1:
        state = 0
        if state == 0:
            state = (1 if x['HUMIDITY']<88.3680573 else 2)
            if state == 1:
                state = (3 if x['HUMIDITY']<78.4797134 else 4)
                if state == 3:
                    state = (7 if x['SUNLIGHT']<3034.30249 else 8)
                    if state == 7:
                        return 3.28497672
                    if state == 8:
                        return 5.02436399
                if state == 4:
                    state = (9 if x['HUMIDITY']<85.9721146 else 10)
                    if state == 9:
                        return 7.12607288
                    if state == 10:
                        return 6.15104771
            if state == 2:
                state = (5 if x['SUNLIGHT']<4113.55762 else 6)
                if state == 5:
                    state = (11 if x['SUNLIGHT']<3666.51733 else 12)
                    if state == 11:
                        return 7.56090403
                    if state == 12:
                        return 8.7587328
                if state == 6:
                    state = (13 if x['HUMIDITY']<94.1682968 else 14)
                    if state == 13:
                        return 10.9393091
                    if state == 14:
                        return 9.4783268
    elif num_booster == 2:
        state = 0
        if state == 0:
            state = (1 if x['HUMIDITY']<88.2817688 else 2)
            if state == 1:
                state = (3 if x['HUMIDITY']<78.2046661 else 4)
                if state == 3:
                    state = (7 if x['SUNLIGHT']<3026.52051 else 8)
                    if state == 7:
                        return 2.93298054
                    if state == 8:
                        return 4.45767546
                if state == 4:
                    state = (9 if x['SUNLIGHT']<3266.39355 else 10)
                    if state == 9:
                        return 6.56495047
                    if state == 10:
                        return 5.67538786
            if state == 2:
                state = (5 if x['SUNLIGHT']<4128.89258 else 6)
                if state == 5:
                    state = (11 if x['SUNLIGHT']<3661.52124 else 12)
                    if state == 11:
                        return 6.76490164
                    if state == 12:
                        return 7.88719416
                if state == 6:
                    state = (13 if x['HUMIDITY']<94 else 14)
                    if state == 13:
                        return 9.94202709
                    if state == 14:
                        return 8.58524323

def xgb_predict(x):
    predict = 1125.0349232538374
# initialize prediction with base score
    for i in range(3):
        predict = predict + xgb_tree(x, i)
    return predict
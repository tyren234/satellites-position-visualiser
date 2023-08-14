import numpy as np
from datetime import date

def date2tow(data, rollover=False):    
    dday=date.toordinal(date(data[0],data[1],data[2])) - (date.toordinal(date(1980,1,6)))
    week = dday//7
    day = dday%7
    tow = day * 86400 + data[3] * 3600 + data[4] * 60 + data[5]
    if rollover:
        if 0<week<=1023:
            week = week
        elif 1023<week<=2047:
            week = week-2**10
        elif 2047<week<=2**12-1:
            week = week-2**11
    return week, tow


def tow2date(gpstime, rollover=0):
    days = gpstime[0]*7
    dayplus = gpstime[1]//86400
    if rollover==0:
        data2 = date.fromordinal(date.toordinal(date(1980,1,6)) + days + dayplus)
    elif rollover == 1:
        data2 = date.fromordinal(date.toordinal(date(1999,8,21)) + days + dayplus)
    elif rollover == 2:
        data2 = date.fromordinal(date.toordinal(date(2019,4,7)) + days + dayplus)
    hours = (gpstime[1]%86400)//3600
    minutes = (gpstime[1]%86400)//60 - hours*60
    seconds = (gpstime[1]%86400) - hours*3600 - minutes * 60    
    data = [data2.year, data2.month, data2.day, hours, minutes, seconds]
    return data
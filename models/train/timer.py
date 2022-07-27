import time 
def GetTimeByDict(timer):
    output = ""
    for key,value in timer.items():
        output = output + f'{key}: {round(value,3)}s, '
    return output
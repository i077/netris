import numpy as np
data = open('playerdata.dat', 'r')
lnum = 0
states = []
actions = []
for line in data:
    l = line.lstrip('[').rstrip(']\n')
    sp = l.split(', ')
    #print(sp)
    #break
    if lnum % 2 == 0:
        s = list(map(float, sp))
        states.append(s)
    elif lnum % 2 == 1:
        a = list(map(float, sp))
        actions.append(a)
    lnum += 1
print(states[0], actions[0])

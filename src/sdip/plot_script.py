"""
Lecture of log2.log line to line, then plot a figure using matplot.pyplot
"""

import matplotlib.pyplot as plt

file_object = open("/home/turing/Repos/openai/src/sdip/log2.xor");

timesteps = []
for line in file_object: 
    if line[0] != " " and line[0] != "[":
        line = line.strip()
        timesteps += [int(line)]
file_object.close()
plt.plot(timesteps)
plt.axis([0, 55000, 0, 200])
plt.show()

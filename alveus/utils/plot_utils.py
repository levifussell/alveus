import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class LiveDataGraph:

    def __init__(self, data_function, update_rate=200, title=""):
        self.data_function = data_function
        x_data, y_data = self.data_function()
        self.figure = plt.figure(figsize=(4, 4))
        self.figure.suptitle(title)
        self.line, = plt.plot(x_data, y_data)
        self.update_rate = update_rate
        self.animation = FuncAnimation(self.figure, self.__update__,
                                       interval=self.update_rate)

    def __update__(self, frame):
        x_data, y_data = self.data_function()
        self.line.set_data(x_data, y_data)
        self.figure.gca().relim()
        self.figure.gca().autoscale_view()
        return self.line

    def run(self):
        plt.show(block=False)
        plt.pause(0.001)


class LivePlotHistogram:
    def __init__(self, data_functions, names, update_rate=1, barCount=50,
                 title=""):
        self.data_functions = data_functions
        self.names = names
        self.figure, self.axs = plt.subplots(len(self.data_functions), 1,
                                             figsize=(4, 4))
        self.figure.suptitle(title)
        self.title = title
        self.update_rate = update_rate
        self.barCount = barCount
        self.centres = []
        self.widths = []
        self.hists = []
        self.timestep = 0

    def __update__(self):
        self.centres = []
        self.widths = []
        self.hists = []
        for i in self.data_functions:
            y_data = i()
            h, bars = np.histogram(y_data, bins=self.barCount)
            c = (bars[1:] + bars[:-1]) / 2.0
            w = (bars[0] - bars[1]) * 0.8
            self.centres.append(c)
            self.widths.append(w)
            self.hists.append(h)

    def run(self):
        if self.timestep % self.update_rate == 0:
            self.__update__()
            for idx, (h, c, w) in enumerate(zip(self.hists, self.centres,
                                                self.widths)):
                self.axs[idx].clear()
                self.axs[idx].bar(c, h, width=w)
                self.axs[idx].set_title(self.names[idx])

            plt.show(block=False)
            plt.pause(0.001)

        self.timestep += 1

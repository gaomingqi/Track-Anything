import time

class Timer:
    def __init__(self):
        self._acc_time = 0
        self._paused = True

    def start(self):
        if self._paused:
            self.last_time = time.time()
            self._paused = False
        return self

    def pause(self):
        self.count()
        self._paused = True
        return self

    def count(self):
        if self._paused:
            return self._acc_time
        t = time.time()
        self._acc_time += t - self.last_time
        self.last_time = t
        return self._acc_time

    def format(self):
        # count = int(self.count()*100)
        # return '%02d:%02d:%02d' % (count//6000, (count//100)%60, count%100)
        return '%03.2f' % self.count()

    def __str__(self):
        return self.format()
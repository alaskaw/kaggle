import time
import datetime

class Timer(object):
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.t0 = time.time()
        
    def __exit__(self, *args):
        print "\nRan " + self.name + " in " + str(datetime.timedelta(seconds=time.time()-self.t0)) + "\n"
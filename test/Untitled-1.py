import numpy as np
from threading import Thread
import time
def test(d=1):
    a.append(d)

start = time.time()
a = []
t = []

t.append(Thread(target = test, args=(1, )))
print(t[-1])
t2 = Thread(target = test, args=(2, ))
t2.start()
t3 = Thread(target = test, args=(3, ))
t3.start()
print(a)

print(t[-1])

stop = time.time() - start
print(stop, time.strftime("%H:%M:%S", time.gmtime(stop)))


start = time.time()
a = []
test(1)
test(2)
test(3)
print(a)
stop = time.time() - start
print(stop, time.strftime("%H:%M:%S", time.gmtime(stop)))
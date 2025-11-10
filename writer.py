import time
import numpy as np

while True:
    p = np.random.randint(4)
    for i in range(10):
        with open('output.txt', 'a') as file:            
            file.write(f'{p} {time.time()}\n')
        print(p)
        time.sleep(1)

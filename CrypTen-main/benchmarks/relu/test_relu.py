import numpy as np
import random

def relu(x):
    if x > 0:
        return x
    else:
        return 0
    
def custom_mod(a, b):
    # 模拟c++中的mod运算，即余数符号与a相同
    result = a % b
    if a < 0:
        result -= b
    return result
    
def test_unit(x):
    R = random.randint(1, 1000)
    R_rec = 1 / R
    
    y = custom_mod(x * R, 13)
    if y == 0:
        y += 1
        tag = 1
    else:
        tag = 0
    a = int((x * R) / 13)
    
    z = relu(y)
    r = z * (13*a / y) + z - tag
    r = r * R_rec
    
    return r

x = random.randint(-1000, 1000)

r1 = relu(x)
r2 = test_unit(x)

print("x: {}, real:{}, appr: {}".format(x, r1, r2))
import numpy as np

def space(n):
    Ad = 4.11
    fc = 9.15
    e = 2.8
    di = np.linspace(3.5,5,n)
    hi = Ad*(3/(4*3.1415*9.15*di))**e
    return hi
    
def random_space(n,i):
    np.random.seed(i)
    Ad = 4.11
    fc = 9.15
    e = 2.8
    di = np.random.rand(n)*1.5+3.5
    hi = Ad*(3/(4*3.1415*9.15*di))**e
    return hi
    
if __name__ == "__main__":
    k = 5
    for i in range(5):
        h = random_space(k,i)
        print(h)
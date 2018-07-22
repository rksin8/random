import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm


def J(x):
    t = np.array([[8*x[0], -3*x[1]**2],[9*x[0]**2, 8*x[1]]])
    return t

def J_inv(x):
    return inv(J(x))


def R(x):
    R1 = 4*x[0]**2 -x[1]**3 + 28
    R2 = 3*x[0]**3 + 4*x[1]**2 - 145 

    return np.array([R1,R2]).T

def main():
    x = np.array([1,1]) # initial guess x1=1, x2=1
    l2_norm = norm(R(x))
    print(l2_norm)
    max_itr = 1;

    print("No. of Iterations \t l2_norm")

    while l2_norm >0.000001 and max_itr<100:
        x1n = x- np.dot(J_inv(x),R(x))

        x = x1n
        l2_norm = norm(R(x));
        print("%2d\t\t\t %0.3f" %(max_itr, l2_norm))
        max_itr = max_itr +1
        #print(l2_norm,R(x))



    print("Solution: ", x)
    print("Function value: ",R(x))




if __name__ == "__main__":
    main()


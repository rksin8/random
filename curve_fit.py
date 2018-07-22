import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x,a,b,c):
    return a* np.exp(-b*x)+c


def main():
    x = np.linspace(0,4,50)
    y = func(x, 2.5,1.3,0.5)

    np.random.seed(1729)
    y_noise = 0.2 *np.random.normal(size=x.size)

    y = y+ y_noise

    plt.plot(x,y,'b-',label='data')
    popt,pcov = curve_fit(func,x,y)
    plt.plot(x,func(x,*popt), 'r-', label='fit: a=%3.5f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()

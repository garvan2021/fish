
## Forward Difference ##
#def numerical_diff(f, x):
#    h = 10e-50
#    return (f(x+h) - f(x)) / h

## Central Difference ##
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

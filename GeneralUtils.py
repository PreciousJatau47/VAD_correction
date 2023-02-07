import numpy as np

# Note: Numpy's logical_and works for different length inputs
def logical_and(*argv):
    if not argv:
        return None
    result = np.array(argv[0])
    for arg in argv:
        result = np.logical_and(result, arg)
    return result


# def Main():
#     ans = logical_and([True], [True, True], [True])
#     print(ans)
#
# Main()
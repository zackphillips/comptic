# This file contains functions useful for assessing error between two measurements

def sse(A,B):
    ''' Sum of squared errors '''
    s = 0
    for i in  range(3):
        s += sum((A[:,:,i] - B[:,:,i])*A[:,:,i] - B[:,:,i])
    return s

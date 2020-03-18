"""
    A class implementing an n-dimensional iterator for being able to iterate trough arrays

    When creating you can either give an object or the ROI coordinates min_value and max_value as ists
    by choosing exclude_axes, this axis is replaced by a range in the iterator.
"""
import numpy as np

class ndIterator:
    def __init__(self, min_value, max_value=None, exclude_axes = None):
        if isinstance(min_value, np.ndarray):
            max_value = min_value.shape
            min_value = np.zeros(min_value.ndim,dtype=int)
        if len(min_value) != len(max_value):
            raise('Error using ndIterator: min_value and max_value have to match in size')
        self.current = list(min_value).copy()
        self.low = self.current.copy()
        self.high = list(max_value).copy()
        if exclude_axes is not None: # the user wants to iterate only over these axes
            for d in [exclude_axes]:
                self.low[d] = slice(0,self.high[d],None)
                self.current[d] = slice(0,self.high[d],None)

    def __iter__(self):
        return self

    def __next__(self):
        for d in range(len(self.current)):
            if not isinstance(self.current[d], slice):
                if self.current[d] >= self.high[d]:
                    raise StopIteration
        else:
            prev = tuple(self.current) # .copy()
            for d in range(len(self.current)):
                ind = -d-1
                if isinstance(self.current[ind], slice):
                    if d < len(self.current)-1:
                        continue # do nothing for this dimension
                    else:
                        self.current[ind] = 1
                        self.high[ind] = 1 # forces the iterations to end in the next call
                self.current[ind] += 1
                if self.current[ind] >= self.high[ind]:
                    if d < len(self.current)-1:
                        self.current[ind] = self.low[ind]
                    else:
                        return prev # one last time
                else:
                    return prev
            return prev

    def relPos(self, current):
        """
        get the relative position in the ROI (having slices not counting)
        :return: the index without the slices
        """
        relpos = list()
        for pos,high,low in zip(current,self.high,self.low):
            if not isinstance(pos,slice):
                myrange = high-low
                mymid = low + myrange//2
                relpos.append((pos-mymid) / myrange)
        return np.array(relpos)

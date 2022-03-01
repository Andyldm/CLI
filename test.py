import numpy  as np

s=[1, 2, 3]
k=1
s_filter = np.array(
                [k in lst for lst in s]
            )if True else s == k
print(s_filter)
import numpy as np

a = np.array([2, 3, 4])
b = np.array([(15, 2, 3),(4, 5, 6)])
x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
c = np.arange(15).reshape(3, 5)
d = np.zeros((3, 4))
e = np.ones((2, 3, 4), dtype=np.int16)

# print('this is A :',a,'\nthis is B :', b,'\nthis is C :', c,'\nthis is D :' ,d,'\nthis is E :', e)

# print(b[0][0])
print(x)
print(x.ndim) # บอกมิติของ Array
print(x[0][0][1])
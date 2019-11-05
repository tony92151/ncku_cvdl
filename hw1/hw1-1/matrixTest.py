import matplotlib as plt
import numpy as np

k = np.array([[1,4,2],[3,1,5],[7,1,2]])
r = np.array([[1,4,3,1],[2,1,2,1],[1,1,9,2]])

E_kp = np.concatenate((k@r, np.array([[0,0,0,1]])),axis = 0)

x = np.array([[4],[7],[1],[1]])

p = E_kp@x

print("E_kp @ X = p")

print("E_kp:\n",E_kp)

print("X:\n",x)

print("p\n",E_kp@x)

print("o\n",k@r@x)

print("================================")

print("E_kp^-1 @ p = x")

print("E_kp inv:\n",np.linalg.inv(E_kp))

print("p:\n",p)

print("x:\n",np.linalg.inv(E_kp) @ p)
# np.linalg.inv(a)
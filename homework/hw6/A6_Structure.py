import numpy as np
from scipy.optimize import linprog

#potentials
f = np.array([1., 0., 0., 1., 0., 5., 0., 0.])

#local probability constraints
##############################
## specify the constraints which have a value of one as their right hand side
## Dimensions: A_eq1 (list of three lists each with 8 entries)
##############################
A_eq1 = [[, , , , , , , ],[, , , , , , , ],[, , , , , , , ]]
b_eq1 = [[1],]*3

#marginalization constraints
##############################
## specify the constraints which have a value of zero as their right hand side
## Dimensions: A_eq2 (list of three lists each with 8 entries)
##############################
A_eq2 = [[, , , , , , , ],[, , , , , , , ],[, , , , , , , ],[, , , , , , , ]]
b_eq2 = [[0],]*4

#bounds
bounds = [(0, 1),]*8

res = linprog(-f, A_eq=np.concatenate((A_eq1,A_eq2)), b_eq=np.concatenate((b_eq1,b_eq2)), bounds=bounds)
print(res)

print(np.matmul(A_eq1, res.x))
print(np.matmul(A_eq2, res.x))
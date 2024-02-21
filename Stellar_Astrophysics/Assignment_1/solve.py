import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy
import pdb

#constants
rsol = 6.957e10 #CGS
msol = 1.989e33 #CGS
lsol = 3.8e33   #cgs

G = 6.67e-8 #CGS
e_charge = 4.8e-10 #CGS electron charge
kb = 1.38e-16 #CGS boltzman constant
Na = 6.02e23 #advagadros number
amu = 1.66e-24 #g atomic mass unit
h = 6.63e-27  #planks constant cgs
me = 9.1e-28 #electron mass grams
c = 3e10 # speed of light CGS
rad_constant = 7.56e-15 #CGS radiation constant


ue = amu*Na*(0.715+0.5*0.271+0.5*0.014)**-1
u1 = 1.28
u = (1/u1 + 1/ue)**-1
print('ue =', ue, 'ui =', u1, 'u =', u  )
# this throws an error: no algorithms are implemented to solve equation

import numpy as np
import para_fun as P
num_point=1000000
the=np.linspace(P.a, P.b, int(num_point))
real=P.give_real(the)
d_real=P.give_d_real(the)
dd_real=P.give_dd_real(the)
J=1/2*P.miu*dd_real**2+P.ro*real
print((P.b-P.a)*np.mean(J))
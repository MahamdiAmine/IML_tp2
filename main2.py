from oct2py import Oct2Py

k = 5
p = 0.2

with Oct2Py() as oc:
    oc.eval('pkg load signal')
    bmat = oc.fir1(k, p)
print(bmat)
# %%
import scipy.signal

bpy = scipy.signal.firwin(k + 1, p)
print(bpy)

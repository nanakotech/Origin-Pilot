import pyqpanda as pq
import numpy as np

#
# def f(a=1, b=2):
#     return a+b
#
# def f1(a):
#     return f(a, b=2)
#
# def g(func):
#     res = func(3)
#     return res
#
#
# print(g(f1))

def create_cir(qlist):
    cir = pq.QCircuit()
    cir << pq.RY(qlist[0], np.pi / 3) << pq.X(qlist[1]).control(qlist[0])
    return cir


m = pq.CPUQVM()
m.initQVM()
q_state = m.qAlloc_many(2)

prog = pq.QProg()
prog << create_cir(q_state)

print(prog)
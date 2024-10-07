import sympy as sp
import torch

l0 = [-5.1979e-05, 0.0233, -0.033]
l1 = [0, 0.0568, 0]
l2 = [0, 0, -0.06925]
l3 = [0, 0, -0.07025]
l4 = [0, 0, -0.14]
l5 = [0.07525, 0, 0]
j0, j1, j2, j3, j4, j5 = sp.symbols("j0 j1 j2 j3 j4 j5")


def return_rt_mat_p(j, l):
    return sp.Matrix(
        [
            [sp.cos(j), 0, sp.sin(j), l[0]],
            [0, 1, 0, l[1]],
            [-sp.sin(j), 0, sp.cos(j), l[2]],
            [0, 0, 0, 1],
        ]
    )


def return_rt_mat_r(j, l):
    return sp.Matrix(
        [
            [1, 0, 0, l[0]],
            [0, sp.cos(j), -sp.sin(j), l[1]],
            [0, sp.sin(j), sp.cos(j), l[2]],
            [0, 0, 0, 1],
        ]
    )


def return_rt_mat_y(j, l):
    return sp.Matrix(
        [
            [sp.cos(j), -sp.sin(j), 0, l[0]],
            [sp.sin(j), sp.cos(j), 0, l[1]],
            [0, 0, 1, l[2]],
            [0, 0, 0, 1],
        ]
    )


ll = sp.Matrix(
    [
        [0],
        [0],
        [0],
        [1],
    ]
)
j_0_rt = return_rt_mat_p(j0, l0)
j_1_rt = return_rt_mat_r(j1, l1)
j_2_rt = return_rt_mat_y(j2, l2)
j_3_rt = return_rt_mat_p(j3, l3)
j_4_rt = return_rt_mat_p(j4, l4)
j_5_rt = return_rt_mat_r(j5, l5)
rt = j_0_rt 
rt = rt.subs({j1: 0, j2: 0, j5: 0})
sp.pprint(sp.trigsimp(rt))
rt = j_0_rt * j_1_rt 
rt = rt.subs({j1: 0, j2: 0, j5: 0})
sp.pprint(sp.trigsimp(rt))
rt = j_0_rt * j_1_rt * j_2_rt  
rt = rt.subs({j1: 0, j2: 0, j5: 0})
sp.pprint(sp.trigsimp(rt))
rt = j_0_rt * j_1_rt * j_2_rt * j_3_rt 
rt = rt.subs({j1: 0, j2: 0, j5: 0})
sp.pprint(sp.trigsimp(rt))
rt = j_0_rt * j_1_rt * j_2_rt * j_3_rt * j_4_rt
rt = rt.subs({j1: 0, j2: 0, j5: 0})
sp.pprint(sp.trigsimp(rt))

rt = j_0_rt * j_1_rt * j_2_rt * j_3_rt * j_4_rt * j_5_rt
# 打印结果
# print("Matrix:")
# sp.pprint(ll)
rt = rt.subs({j1: 0, j2: 0, j5: 0})
# sp.pprint(rt)
sp.pprint(sp.trigsimp(rt))

rt = rt.subs({j0: 0, j1: 0, j2: 0, j3: 0, j4: 0, j5: 0})
# sp.pprint(rt)
sp.pprint(sp.trigsimp(rt))


rt = (
    return_rt_mat_y(j0, [0, 0, 0])
    * return_rt_mat_p(j1, [0, 0, 0])
    * return_rt_mat_r(j2, [0, 0, 0])
)
# sp.pprint(rt)
# sp.pprint(sp.trigsimp(rt))
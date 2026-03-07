# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
from pyqpanda3.core import CPUQVM, QCircuit, QProg, CNOT, X, RY

from .. plugin import *

def plot_bar(dic):
    """
    Plot a bar chart from a dictionary.

    Highlights the bar with the highest value in red, while the rest are shown in blue.

    Parameters
    ----------
    dic : dict
        A dictionary where keys are labels and values are numerical quantities to be plotted.
    """
    keys = list(dic.keys())
    values = list(dic.values())
    max_value_index = values.index(max(values))
    colors = ['red' if i == max_value_index else 'blue' for i in range(len(keys))]
    plt.bar(keys, values, color=colors)
    plt.show()
    plt.clf()


def plot_loss(los):
    """
    Plot a line chart showing the loss over iterations.

    Useful for visualizing the optimization process.

    Parameters
    ----------
    los : list of float
        A list of loss values, where each entry corresponds to one optimization step.
    """
    index = list(range(1, 1 + len(los)))
    plt.plot(index, los)
    plt.show()
    plt.clf()

def control(qubits1, qubits2, theta):

    """
    Construct a multi-controlled RY rotation gate.

    Creates an RY rotation gate on the target qubit with multiple control qubits.

    Parameters
    ----------
    qubits1 : list or Qubit
        Control qubit(s). Can be a single Qubit or a list of Qubits.
    qubits2 : Qubit
        Target qubit on which the RY rotation will be applied.
    theta : float
        Rotation angle for the RY gate.

    Returns
    -------
    QGate
        A controlled RY gate that can be appended to a quantum circuit.
    """
    qvec = qubits1
    RY_control = RY(qubits2, theta).control(qvec)
    return RY_control

class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) Optimizer.

    This class implements the SPSA algorithm for optimizing a scalar-valued
    objective function that depends on multiple parameters. SPSA is useful
    in cases where the objective function is noisy, non-differentiable, or
    expensive to evaluate, such as in variational quantum algorithms.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        The objective function to be minimized. It must take a parameter vector as input
        and return a scalar loss value.
    a : float, optional
        Scaling factor for the learning rate (step size), default is 0.1.
    c : float, optional
        Scaling factor for the perturbation, default is 0.1.
    alpha : float, optional
        Exponent controlling the decay rate of the learning rate, default is 0.602.
    beta : float, optional
        Exponent controlling the decay rate of the perturbation size, default is 0.101.
    max_iters : int, optional
        Maximum number of iterations to perform, default is 1000.
    """

    def __init__(self, objective_function, a=0.1, c=0.1, alpha=0.602, beta=0.101, max_iters=1000):
        self.objective_function = objective_function
        self.a = a
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.max_iters = max_iters

    def optimize(self, initial_params):
        """
        Run the SPSA optimization to minimize the objective function.

        Parameters
        ----------
        initial_params : np.ndarray
            Initial values for the parameters to be optimized.

        Returns
        -------
        np.ndarray
            The optimized parameter vector.
        list[float]
            History of objective function values over iterations.
        """
        m = len(initial_params)
        history = []
        params = initial_params
        history.append(float(self.objective_function(initial_params.copy())))
        for k in range(1, self.max_iters + 1):
            ak = self.a / (k + 1) ** self.alpha
            ck = self.c / (k + 1) ** self.beta

            delta = np.random.choice([-1, 1], size=m)

            params_plus = params + ck * delta
            params_minus = params - ck * delta

            f_plus = self.objective_function(params_plus)
            f_minus = self.objective_function(params_minus)
            grad_estimate = (f_plus - f_minus) / (2 * ck * delta)

            params = params - ak * grad_estimate
            x = self.objective_function(params.copy())
            history.append(float(x))
        return params, history



class Feature_Selection:
    """
    Quantum-based Feature Selection using parameterized quantum circuits.

    This class builds a variational quantum circuit to model a feature selection
    task. The objective is to optimize the selection of a subset of features
    based on a given linear and quadratic objective function.

    Parameters
    ----------
    quadratic : np.ndarray
        Quadratic coefficient matrix used in the objective function.
    linear : list[float]
        Linear coefficient list used in the objective function.
    select_num : int
        Number of features to be selected (i.e., number of bits set to 1).

    Attributes
    ----------
    qb_num : int
        Number of qubits required, equal to the number of features.
    l1 : list[float]
        History of linear loss values during optimization.
    l2 : list[float]
        History of quadratic loss values during optimization.
        
    >>> from pyqpanda_alg import QmRMR
    >>> import numpy as np
    >>> from pyqpanda3.core import *
    >>> import matplotlib.pyplot as plt
    
    >>> m = 6
    >>> u = np.random.random(m)
    >>> cor = np.random.random([m, m])
    >>> cor = (cor.T + cor)/2
    >>> ini_par = np.random.random(int(m/2)*m)*np.pi
    >>> loss, choice, dic = QmRMR.Feature_Selection(cor, u, 3).get_his_res(ini_par)
    >>> print(choice)
    >>> plt.plot(loss)
    >>> plt.show()
    [0, 1, 0, 0, 1, 1]
    
    """

    def __init__(self, quadratic, linear, select_num):
        self.quadratic = quadratic
        self.linear = linear
        self.qb_num = len(linear)
        self.select_num = select_num
        self.l1 = []
        self.l2 = []

    def Circuit(self, qbs, para):
        cir = QCircuit()
        n = len(qbs)
        for i in range(self.select_num):
            cir << X(qbs[i])

        cz_list_even = [[t, t + 1] for t in range(0, n - 1, 2)]
        cz_list_odd = [[t + 1, t + 2] for t in range(0, n - 2, 2)]
        k = -1
        for i in range(int(n/2)):
            for kf in cz_list_even:
                k += 1
                cir << CNOT(qbs[kf[1]], qbs[kf[0]]) << control(qbs[kf[0]], qbs[kf[1]], para[k]) << CNOT(
                    qbs[kf[1]], qbs[kf[0]])
            for kf in cz_list_odd:
                k += 1
                cir << CNOT(qbs[kf[1]], qbs[kf[0]]) << control(qbs[kf[0]], qbs[kf[1]], para[k]) << CNOT(
                    qbs[kf[1]], qbs[kf[0]])
        return cir


    def select_key(self, dit):
        new_dit = {}
        for key in dit:
            if key.count('1') == self.select_num:
                new_dit[key] = dit[key]
        top_10_items = dict(sorted(new_dit.items(), key=lambda item: item[1], reverse=True)[:10])
        top_10_items = {key: round(value, 3) for key, value in top_10_items.items()}
        return next(iter(top_10_items)), top_10_items

    def get_theory(self, para):
        machine_the = CPUQVM()
        prog = QProg(self.qb_num)
        qv = prog.qubits()
        prog << self.Circuit(qbs=qv, para=para)
        machine_the.run(prog, 1000)
        res = machine_the.result().get_prob_dict(qv)
        res = parse_quantum_result_dict(res, qv, select_max=-1)
        return res


    def cal_loss(self, para):
        machine_the = CPUQVM()
        prog = QProg(self.qb_num)
        qv = prog.qubits()
        prog << self.Circuit(qbs=qv, para=para)
        machine_the.run(prog, 1000)
        dict = machine_the.result().get_prob_dict(qv)
        dict = parse_quantum_result_dict(dict, qv, select_max=-1)
        res = 0
        loss1 = 0
        loss2 = 0
        for i in dict:
            ans_list = np.array([int(c) for c in i])
            loss1 = -ans_list @ self.linear * dict[i] + loss1
            loss2 = (ans_list @ self.quadratic @ ans_list) * dict[i] + loss2
            res = loss2 - loss1

        self.l1.append(loss1)
        self.l2.append(loss2)


        return res


    def get_his_res(self, ini_para):
        np.random.seed(1234)
        optimizer = SPSAOptimizer(self.cal_loss, max_iters=200)
        optimal_params, his = optimizer.optimize(ini_para)
        x = self.get_theory(optimal_params)
        key, dic = self.select_key(x)
        choice = [int(t) for t in key]
        return his, choice, dic

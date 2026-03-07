import pyqpanda as pq
from pyqpanda_alg.QFinance import grover


if __name__ == '__main__':
    m = pq.CPUQVM()
    m.initQVM()
    q_state = m.qAlloc_many(3)

    def mark(qubits):
        return grover.mark_data_reflection(qubits=qubits, mark_data=['101', '001'])


    demo_search = grover.Grover(flip_operator=mark)
    # iter_num = grover.iter_num(q_num=len(q_state), sol_num=2)
    # print('best iter num: ', iter_num)
    # prob, angle = grover.iter_analysis(q_num=len(q_state), sol_num=2, iternum=iter_num)
    # print('prob for getting one of the solution with given iter num:', prob)

    prog = pq.QProg()
    prog << demo_search.cir(q_input=q_state)

    res = m.prob_run_dict(prog, q_state)
    print(res)
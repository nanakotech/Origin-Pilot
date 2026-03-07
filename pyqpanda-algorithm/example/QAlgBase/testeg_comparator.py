from pyqpanda_alg.QFinance import comparator
import pyqpanda as pq


if __name__ == '__main__':
    # 整数比较
    value = 3
    m = pq.CPUQVM()
    m.initQVM()
    q_state = m.qAlloc_many(2)
    q_anc_cmp = m.qAlloc_many(2)
    prog = pq.QProg()
    prog << pq.H(q_state)
    cir = comparator.int_comparator(value, q_state, q_anc_cmp, function='g', reuse=True)
    prog << cir
    res = m.prob_run_dict(prog, [q_anc_cmp[-1]])
    print(res)

    # 插值方法
    value = 3.3
    m = pq.CPUQVM()
    m.initQVM()
    q_state = m.qAlloc_many(3)
    q_anc_cmp = m.qAlloc_many(3)
    prog = pq.QProg()
    prog << pq.X(q_state[:2])
    cir = comparator.interpolation_comparator(value, q_state, q_anc_cmp, function='g', reuse=True)
    prog << cir
    res = m.prob_run_dict(prog, [q_anc_cmp[-1]])
    print(res)

    # 两个态比较，示例中叠加态的0，1，2，3有0.5的概率大于态1
    m = pq.CPUQVM()
    m.initQVM()
    q_state_1 = m.qAlloc_many(2)
    q_state_2 = m.qAlloc_many(2)
    q_anc_cmp = m.qAlloc_many(2)
    prog = pq.QProg()
    prog << pq.H(q_state_1)
    prog << pq.X(q_state_2[0])
    cir = comparator.qubit_comparator(q_state_1, q_state_2, q_anc_cmp, function='g')
    prog << cir
    res = m.prob_run_dict(prog, [q_anc_cmp[-1]])
    print(res)
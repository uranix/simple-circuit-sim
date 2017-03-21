import numpy as _np
from scipy.sparse import csr_matrix, csgraph
from assimulo.problem import Implicit_Problem
from assimulo.solvers.sundials import IDA

class Terminal(object):
    def __init__(self, elem, name):
        self.elem = elem
        self.name = name
    def assign(self, tid):
        self.tid = tid
    def plug(self, node):
        self.node = node
    def __repr__(self):
        return self.name + ' of ' + repr(self.elem)
    def element(self):
        return self.elem

class State(object):
    def __init__(self, elem, name, value):
        self.elem = elem
        self.name = name
        self.val = value
    def assign(self, sid):
        self.sid = sid
    def __repr__(self):
        return self.name + ' of ' + repr(self.elem)
    def element(self):
        return self.elem

from elements import NotConnected

class Circuit(Implicit_Problem):
    def __init__(self, name=None):
        self.connections = set()
        self.terminals = set()
        self.connected_terminals = set()
        self.elements = set()
        self.states = []
        self.name = name if name else repr(self)

    def add_element(self, elem):
        if elem in self.elements:
            return
        self.elements.add(elem)
        self.terminals |= set(elem.terminals)
        self.states += elem.states

    def connect(self, *args):
        for term in args:
            self.add_element(term.element())
            self.connected_terminals.add(term)
        for i in range(1, len(args)):
            self.connections.add((args[i-1], args[i]))

    def process_not_connected(self):
        """
        Attach a NotConnected element to each not connected terminal
        """
        terminals_to_connect = self.terminals - self.connected_terminals
        for i,term in enumerate(terminals_to_connect):
            nc = NotConnected('NC[auto]' + str(i))
            self.connect(term, nc[0])
        assert self.connected_terminals == self.terminals

    def enumerate_terminals(self):
        ordered_terminals = []
        for i,term in enumerate(self.terminals):
            term.assign(i)
            ordered_terminals.append(term)
        self.ordered_terminals = ordered_terminals

    def enumerate_states(self):
        for i,state in enumerate(self.states):
            state.assign(i)

    def build_connection_graph(self):
        edges = _np.empty((2, len(self.connections)), dtype=_np.int)
        for i,(t1,t2) in enumerate(self.connections):
            edges[:, i] = t1.tid, t2.tid
        vals = _np.ones(edges.shape[1])
        nterm = len(self.terminals)
        return csr_matrix((vals, edges), shape=(nterm, nterm))

    def generate_nodes(self):
        """
        Given list of connected terminals
        [(t1, t2), (t1, t5), (t2, t3), (t4, t6), ...]
        generate list of nodes

        node  attached terminals
        0     set(t1, t2, t3, t5)
        1     set(t4, t6)
        ...
        """
        self.process_not_connected()
        self.enumerate_terminals()
        self.enumerate_states()

        graph = self.build_connection_graph()
        nodes_num, labels = csgraph.connected_components(graph, directed=False)
        nodes = [set() for _ in range(nodes_num)]
        for i,label in enumerate(labels):
            nodes[label].add(self.ordered_terminals[i])
        for i,node in enumerate(nodes):
            for term in node:
                term.plug(i)
        self.nodes = nodes

    def assemble(self, verbose=True):
        self.generate_nodes()
        if verbose:
            print('Nodes:', len(self.nodes))
            print('Terminals:', len(self.terminals))
            print('States:', len(self.states))
            for i, node in enumerate(self.nodes):
                print('Var ', i, ': Node', i, 'potential (V)')
                for t in node:
                    print('\t *', t)
            for i,term in enumerate(self.terminals):
                print('Var ', i+len(self.nodes), ': Current through', term, '(A)')
            for i,state in enumerate(self.states):
                print('Var ', i+len(self.nodes)+len(self.terminals), ':', state)
        nvars = len(self.nodes) + len(self.terminals) + len(self.states)

        self.t0 = 0.
        self.y0 = _np.zeros(nvars)
        self.yd0 = _np.zeros(nvars)
        for i,state in enumerate(self.states):
            self.y0[i+len(self.nodes)+len(self.terminals)] = state.val

        self.algvar = self.get_algebraic_variables()

    def update_terminals_and_states(self, y, yd):
        ncnt = len(self.nodes)
        nterm = len(self.terminals)

        Vs    = y [:ncnt]
        Is    = y [ncnt:(ncnt+nterm)]
        Qs    = y [(ncnt+nterm):]
        dQsdt = yd[(ncnt+nterm):]

        # Set node potential to every attached terminal
        for i,node in enumerate(self.nodes):
            for term in node:
                term.V = Vs[i]

        for i,term in enumerate(self.terminals):
            term.I = Is[i]

        # Distribute state vector
        for i,state in enumerate(self.states):
            state.val = Qs[i]
            state.der = dQsdt[i]

        return Vs, Is, Qs, dQsdt

    def res(self, t, y, yd):
        Vs, Is, Qs, dQsdt = self.update_terminals_and_states(y, yd)

        r = []
        # Kirchhoff rule for each node:
        # sum of currents = 0
        for node in self.nodes:
            Isum = 0
            for term in node:
                Isum += Is[term.tid]
            r.append(Isum)

        # Account for element laws
        for elem in self.elements:
            r += list(elem.law(t))

        assert len(r) == len(y)
        return _np.array(r)

    def jac(self, c, t, y, yd):
        Vs, Is, Qs, dQsdt = self.update_terminals_and_states(y, yd)

        r = _np.zeros((len(y), len(y)))
        ncnt = len(Vs)
        nterm = len(Is)
        row = 0
        for node in self.nodes:
            for term in node:
                r[row, ncnt + term.tid] = 1
            row += 1

        for elem in self.elements:
            jac_V, jac_I, jac_Q = elem.jac(t, c)
            erows = jac_V.shape[0]
            for j,term in enumerate(elem.terminals):
                # if several terminals of the same element are connected to
                # the same node, sum all their jac_V
                r[row:row+erows, term.node] += jac_V[:, j]
                r[row:row+erows, ncnt + term.tid] = jac_I[:, j]
            for j,state in enumerate(elem.states):
                r[row:row+erows, ncnt + nterm + state.sid] = jac_Q[:, j]
            row += erows
        return r

    def get_algebraic_variables(self):
        r = _np.zeros(len(self.y0))
        # Only state variables are differential
        if len(self.states) > 0:
            r[-len(self.states):] = 1.
        return r

    def test_jacobian(self, dermul=1., h=1e-4):
        n = len(self.y0)
        e = _np.eye(n)

        t0 = self.t0
        y0 = _np.random.rand(len(self.y0))
        yd0 = _np.random.rand(len(self.y0))

        jex = self.jac(dermul, t0, y0, yd0)
        jnum = _np.empty((n, n))

        for i in range(n):
            jnum[:, i] = (self.res(t0, y0+h*e[i], yd0)-self.res(t0, y0-h*e[i], yd0))/(2*h)
            jnum[:, i] += dermul * \
                (self.res(t0, y0, yd0+h*e[i])-self.res(t0, y0, yd0-h*e[i]))/(2*h)
        print('Exact jac\n', jex)
        print('Numerical jac\n', jnum)
        print('Difference\n', _np.linalg.norm(jnum-jex))
        return jex, jnum

    def simulate(self, tmax, nout=500, maxsteps=5000, rtol=1e-8, verbose=False, test_jac_args=None):
        if not hasattr(self, 't0'):
            self.assemble(verbose)
        if test_jac_args is not None:
            self.test_jacobian(*test_jac_args)
        sim = IDA(self)
        flag, _, _ = sim.make_consistent('IDA_YA_YDP_INIT')
        translation = {0:'SUCCESS', 1:'TSTOP_RETURN', 2:'ROOT_RETURN',
            99:'WARNING', -1 :'TOO_MUCH_WORK', -2 :'TOO_MUCH_ACC',
            -3 :'ERR_FAIL', -4 :'CONV_FAIL', -5 :'LINIT_FAIL',
            -6 :'LSETUP_FAIL', -7 :'LSOLVE_FAIL', -8 :'RES_FAIL',
            -9 :'REP_RES_ERR', -10:'RTFUNC_FAIL', -11:'CONSTR_FAIL',
            -12:'FIRST_RES_FAIL', -13:'LINESEARCH_FAIL', -14:'NO_RECOVERY',
            -20:'MEM_NULL', -21:'MEM_FAIL', -22:'ILL_INPUT', -23:'NO_MALLOC',
            -24:'BAD_EWT', -25:'BAD_K', -26:'BAD_T', -27:'BAD_DKY'}
        if flag < 0:
            raise ArithmeticError('make_consistent failed with flag = IDA_%s' % translation[flag])
        if flag != 0:
            warn('make_consistent returned IDA_%s' % translation[flag])
        sim.rtol = rtol
        sim.maxsteps = maxsteps
        T,Y,Yd = sim.simulate(tmax, nout)
        ncnt  = len(self.nodes)
        nterm = len(self.terminals)

        self.T = T
        Vs    = Y [:, :ncnt]
        dVsdt = Yd[:, :ncnt]
        Is    = Y [:, ncnt:(ncnt+nterm)]
        dIsdt = Yd[:, ncnt:(ncnt+nterm)]
        Qs    = Y [:, (ncnt+nterm):]
        dQsdt = Yd[:, (ncnt+nterm):]

        for i,node in enumerate(self.nodes):
            for term in node:
                term.V    = Vs[:, i]
                term.dVdt = dVsdt[:, i]

        for i,term in enumerate(self.terminals):
            term.I    = Is[:, i]
            term.dIdt = dIsdt[:, i]

        for i,state in enumerate(self.states):
            state.val = Qs[:, i]
            state.der = dQsdt[:, i]

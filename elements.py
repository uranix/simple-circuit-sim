import numpy as _np

from core import Terminal, State

class Element(object):
    def __init__(self, name, *terminal_names, **init_state):
        self.name = name
        self.terminals = [Terminal(self, name) for name in terminal_names]
        self.states = [State(self, name, val) for name,val in init_state.items()]
        for term in self.terminals:
            setattr(self, term.name, term)
        for state in self.states:
            setattr(self, state.name, state)
    def __getitem__(self, num):
        return self.terminals[num]
    def state(self, num):
        return self.states[num]

class Emf(Element):
    def __init__(self, name=None, E=5, r=0):
        self.r = r
        self.E = E
        super(Emf, self).__init__(name, 'Minus', 'Plus')
    def __repr__(self):
        title = 'EMF ' + self.name if self.name else 'Unnamed EMF'
        return title + ' with r=%g Ohm, E=%g V' % (self.r, self.E)
    def law(self, t):
        E = self.E
        eq1 = self[1].V - self[0].V - E + self[0].I * self.r
        eq2 = self[0].I + self[1].I
        return _np.array([eq1, eq2])
    def jac(self, t, dermul):
        jac_V = _np.array([[-1, 1], [0, 0]])
        jac_I = _np.array([[self.r, 0], [1, 1]])
        jac_Q = _np.zeros((2, 0))
        return jac_V, jac_I, jac_Q

class Resistor(Element):
    def __init__(self, name=None, R=100):
        self.R = R
        super(Resistor, self).__init__(name, 'Pin0', 'Pin1')
    def __repr__(self):
        title = 'Resistor ' + self.name if self.name else 'Unnamed resistor'
        return title + ' with R=%g Ohm' % self.R
    def law(self, t):
        eq1 = self[1].V - self[0].V + self[0].I * self.R
        eq2 = self[0].I + self[1].I
        return _np.array([eq1, eq2])
    def jac(self, t, dermul):
        jac_V = _np.array([[-1, 1], [0, 0]])
        jac_I = _np.array([[self.R, 0], [1, 1]])
        jac_Q = _np.zeros((2, 0))
        return jac_V, jac_I, jac_Q

class Capacitor(Element):
    def __init__(self, name=None, C=1e-5, U0=0):
        self.C = C
        super(Capacitor, self).__init__(name, 'Pin0', 'Pin1', Voltage=U0)
    def __repr__(self):
        title = 'Capacitor ' + self.name if self.name else 'Unnamed capacitor'
        return title + ' with C=%g F' % self.C
    def law(self, t):
        eq1 = self[0].I + self.C * self.Voltage.der
        eq2 = self[0].I + self[1].I
        eq3 = self[1].V - self[0].V - self.Voltage.val
        return _np.array([eq1, eq2, eq3])
    def jac(self, t, dermul):
        jac_V = _np.array([[0, 0], [0, 0], [-1, 1]])
        jac_I = _np.array([[1, 0], [1, 1], [0, 0]])
        jac_Q = _np.array([[dermul*self.C], [0], [-1]])
        return jac_V, jac_I, jac_Q

class Inductor(Element):
    def __init__(self, name=None, L=1e-3, I0=0):
        self.L = L
        super(Inductor, self).__init__(name, 'Pin0', 'Pin1', Current=I0)
    def __repr__(self):
        title = 'Inductor ' + self.name if self.name else 'Unnamed inductor'
        return title + ' with L=%g H' % self.L
    def law(self, t):
        eq1 = self[1].V - self[0].V + self.L * self.Current.der
        eq2 = self[0].I + self[1].I
        eq3 = self[0].I - self.Current.val
        return _np.array([eq1, eq2, eq3])
    def jac(self, t, dermul):
        jac_V = _np.array([[-1, 1], [0, 0], [0, 0]])
        jac_I = _np.array([[0, 0], [1, 1], [1, 0]])
        jac_Q = _np.array([[self.L*dermul], [0], [-1]])
        return jac_V, jac_I, jac_Q

class Ground(Element):
    def __init__(self, name=None):
        super(Ground, self).__init__(name, 'Pin')
    def __repr__(self):
        title = 'Ground ' + self.name if self.name else 'Unnamed ground'
        return title
    def law(self, t):
        eq1 = self[0].V
        return _np.array([eq1])
    def jac(self, t, dermul):
        jac_V = _np.array([[1]])
        jac_I = _np.array([[0]])
        jac_Q = _np.zeros((1, 0))
        return jac_V, jac_I, jac_Q

class NotConnected(Element):
    def __init__(self, name=None):
        super(NotConnected, self).__init__(name, 'Pin')
    def __repr__(self):
        title = 'Not connected terminal ' + self.name if self.name else 'Unnamed NC terminal'
        return title
    def law(self, t):
        eq1 = self[0].I
        return _np.array([eq1])
    def jac(self, t, dermul):
        jac_V = _np.array([[0]])
        jac_I = _np.array([[1]])
        jac_Q = _np.zeros((1, 0))
        return jac_V, jac_I, jac_Q

class Diode(Element):
    def __init__(self, name=None, IS=1e-12, VT=25.85e-3):
        self.IS = IS
        self.VT = VT;
        super(Diode, self).__init__(name, 'Anode', 'Cathode')
    def __repr__(self):
        title = 'Diode ' + self.name if self.name else 'Unnamed diode'
        return title + ' with IS=%g A, VT=%g V' % (self.IS, self.VT)
    def law(self, t):
        VT = self.VT
        VD = self.Anode.V - self.Cathode.V
        # Shockley ideal diode model
        I = self.IS * (_np.exp(VD / VT) - 1)
        eq1 = self.Anode.I - I
        eq2 = self[0].I + self[1].I
        return _np.array([eq1, eq2])
    def jac(self, t, dermul):
        VT = self.VT
        VD = self.Anode.V - self.Cathode.V
        ddVD = -self.IS * _np.exp(VD / VT) / VT
        jac_V = _np.array([[ddVD, -ddVD], [0, 0]])
        jac_I = _np.array([[1, 0], [1, 1]])
        jac_Q = _np.zeros((2, 0))
        return jac_V, jac_I, jac_Q

class Generator(Element):
    def __init__(self, name=None, E=5, freq=50, phase_deg=0, r=0):
        self.r = r
        self.E = E
        self.omega = freq * 2 * _np.pi
        self.phase = phase_deg / 180 * _np.pi
        super(Generator, self).__init__(name, 'Minus', 'Plus')
    def __repr__(self):
        title = 'Generator ' + self.name if self.name else 'Unnamed generator'
        return title + ' with r=%g Ohm, E=%g V, F=%g rad/s, Ph=%g rad' % \
                (self.r, self.E, self.omega, self.phase)
    def law(self, t):
        E = self.E * _np.cos(self.omega*t + self.phase)
        eq1 = self[1].V - self[0].V - E + self[0].I * self.r
        eq2 = self[0].I + self[1].I
        return _np.array([eq1, eq2])
    def jac(self, t, dermul):
        jac_V = _np.array([[-1, 1], [0, 0]])
        jac_I = _np.array([[self.r, 0], [1, 1]])
        jac_Q = _np.zeros((2, 0))
        return jac_V, jac_I, jac_Q

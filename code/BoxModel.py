import numpy as np
import scipy.integrate

class Ode(scipy.integrate.ode):
    """ An interface for ode integration.
    
    This is specialized from scipy.integrate.ode to handle the rk4 and euler
    integration methods developed in EOSC 453.
    
    Args:
        *args: Arguments to pass scipy.integrate.ode
        **kwargs: Keyword arguments to pass scipy.integrate.ode
        
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def set_integrator(self, name, **integrator_params):
        """Define the ode integrator.
        
        Use a special case for the ubc_rk4 and ubc_euler integrators
        developed in EOSC 453. Otherwise, use the standard scipy
        integrators.
        
        Args:
            name: The name of the integrator to use. This can be any
                integrator supported by scipy.integrate.ode or
                'ubc_rk4' or 'ubc_euler'.
            **integrator_params: Integrator params to pass to scipy.integrate.ode
            
        """
        if name == 'ubc_rk4':
            self._ubc_integrator = name
            self.integrate = self._ubc_rk4
        elif name == 'ubc_euler':
            self._ubc_integrator = name
            self.integrate = self._ubc_euler
        else:
            super().set_integrator(name, **integrator_params)
    
    def solve(self, n, tf):
        """Solve the ode system.
        
        Args:
            n (int): The number of time steps.
            tf (number): The stop time.
            
        Returns:
            t (array(number)): An (n+1)-dimensional array containing the time steps and
                initial time.
            Y (array(number)): An (m, n+1)-dimensional array containing the solution.
                m is the number of variables, n is the number of time steps.
                
        """
        Y = np.zeros((n + 1, self.y.size))
        t = np.zeros(n + 1)
        
        # Initial conditions
        Y[0, :] = self.y # Initial condition
        t[0] = self.t
        
        h = (tf - self.t) / n
        for i in range(n):
            y = self.integrate(self.t + h)
            Y[i + 1, :] = y
            t[i + 1] = self.t
        return t, Y
            
    def _ubc_rk4(self, t1):
        # An implementation of RK4 based on Assignment 0 of EOSC 453
        
        # Start point
        t0 = self.t
        y0 = self.y
        
        h = t1 - t0 # Step size
    
        k1 = h * self.f(t0, y0)
        k2 = h * self.f(t0 + h / 2, y0 + k1 / 2)
        k3 = h * self.f(t0 + h / 2, y0 + k2 / 2)
        k4 = h * self.f(t0 + h, y0 + k3)
        
        dy = (k1 + 2 * k2 +  2 * k3 + k4) / 6
        y1 = y0 + dy
        
        self.set_initial_value(y1, t1)
        
        return y1
    
    def _ubc_euler(self, t1):
        # An implementation of the Euler method based on Assignment 0 of EOSC 453
        
        # Start point
        t0 = self.t
        y0 = self.y
        
        h = t1 - t0 # Step size
        
        dy = h * self.f(t0, y0)
        y1 = y0 + dy
        
        self.set_initial_value(y1, t1)
        
        return y1

class BoxModel():
    """ A representation of a box model to define fluxes between components
    in a system.
    
    Args:
        edges (list of edges): A list of weighted directed edges defining the
            connections and rate coefficients for the model. An edge has
            the form (src, dst, k) where src is the source box,
            dst is the destination box and k is the rate coefficient
            for material flowing from src to dst.
        nodes (list of labels, optional): A list of names for the nodes
            in the box model. By default, the names are inferred from
            the edges and given an arbitrary order. If the order of the
            nodes is important, nodes should list the boxes in the desired
            order.
        
    """
    
    def __init__(self, edges, nodes=None):
        self.update_model(edges, nodes)
        self._forcing = dict() # Forcing functions for nodes
            
    @property
    def edges(self):
        """The weighted directed edges of the box model."""
        return self._edges
    
    @property
    def nodes(self):
        """The boxes of the box model."""
        return self._nodes
    
    @property
    def size(self):
        """The number of boxes in the box model."""
        return len(self.nodes)
    
    @property
    def coefficient_matrix(self):
        """The matrix K for the system y_dot = Ky + F."""
        n = self.size
        K = np.zeros((n, n))
        for node, props in self._adjacency.items():
            i = self._nodes.index(node)
            # Add negative coefficients for all the outgoing fluxes on diagonal
            K[i, i] = -1 * np.sum([k[1] for k in props['out'].items()])
            # Add positive coefficients for incoming fluxes
            for src in props['in']:
                j = self._nodes.index(src)
                if node in self._adjacency[src]['out']:
                    k = self._adjacency[src]['out'][node]
                    K[i, j] = k
        return K
    
    @property
    def forcing(self):
        """A dictionary of nodes and their corresponding forcing functions."""
        return self._forcing
    
    @property
    def t(self):
        """The time series for the model ODE problem."""
        return self._t
    
    def update_model(self, edges, nodes=None):
        """Update the box model with new edges.
        
        Args:
            edges (list of edges): A list of weighted directed edges defining the
                connections and rate coefficients for the model. An edge has
                the form (src, dst, k) where src is the source box,
                dst is the destination box and k is the rate coefficient
                for material flowing from src to dst.
            nodes (list of labels, optional): A list of names for the nodes
                in the box model. By default, the names are inferred from
                the edges and given an arbitrary order. If the order of the
                nodes is important, nodes should list the boxes in the desired
                order.

        """
        if nodes is None:
            # Create an arbitrary node ordering
            self._nodes = []
            for edge in edges:
                if edge[0] not in self._nodes:
                    self._nodes.append(edge[0])
                if edge[1] not in self._nodes:
                    self._nodes.append(edge[1])
        else:
            self._nodes = nodes
        self._edges = edges
        self._build_adjacency()
        self._build_ode()
    
    def add_forcing(self, node, f):
        """Add a time-dependent forcing function to a box in the model.
        
        Args:
            node (label): The box to add the forcing function.
            f (function): The forcing function. f should take a single argument for
                time and return corresponding input mass.
        
        """
        self._forcing[node] = f
    
    def clear_forcing(self):
        """Clear all forcing functions defined in the model."""
        for node in self.nodes:
            self._forcing[node] = lambda t: 0.0
        
    def forcing_vector(self, t):
        """The vector of forced mass.
        
        Args:
            t (number): The time the forcing functions should be evaluated at.
        
        Returns:
            An n-dimensional vector of added mass where n is the number of
            boxes in the system.
            
        """
        F = np.zeros(self.size)
        for node, f in self.forcing.items():
            F[self.nodes.index(node)] = f(t)
        return F
    
    def set_integrator(self, name):
        """Define the integrator for the ODE system.
        
        Args:
            name (string): The name of the integrator. Can be any integrator
                supported by Scipy, ubc_rk4 or ubc_euler.
                
        """
        self._integrator = name
        
    def set_initial_value(self, y0, t0):
        """Set the initial values for the ODE system.
        
        Args:
            y0 (array): A vector of initial values for the nodes in the system.
            t0 (number): The initial time.
            
        """
        self._t0 = t0
        self._y0 = y0
    
    def solve(self, n, tf):
        """Solve the ODE system.
        
        Args:
            n (int): The number of time steps for the ODE integrator.
            tf (number): The final time for the model.
            
        
        """
        self._build_ode()
        self._ode.set_integrator(self._integrator)
        self._ode.set_initial_value(self._y0, self._t0)
        t, Y = self._ode.solve(n, tf)
        self._t = t
        self._Y = Y
        
    def solution(self, name):
        """Return the solution for a node in the box model.
        
        Args:
            name (string): The name of the node.
            
        """
        i = self.nodes.index(name)
        return self._Y[:, i]
        
    def plot(self, ax, name,  **kwargs):
        """Plot the solution for a node in the box model.
        
        Args:
            ax: A matplotlib ax object to add the plot to.
            name (string): The name of the node.
            **kwargs: Keyword arguments for matplotlib axe plotting.
        
        """
        ax.plot(self.t, self.solution(name), **kwargs)
        
    def _build_ode(self):
        # Ode system
        K = self.coefficient_matrix
        f = lambda t, y: np.dot(K, y) + self.forcing_vector(t)
        self._ode = Ode(f)
        
    def _build_adjacency(self):
        # Store the box model as a graph with an adjacency list.
        _adjacency = {}
        for src, dst, k in self.edges:
            if src not in _adjacency:
                _adjacency[src] = {
                    'out': {dst: k},
                    'in': []
                }
            else:
                _adjacency[src]['out'][dst] = k
            if dst not in _adjacency:
                _adjacency[dst] = {
                    'out': {},
                    'in': [src]
                }
            else:
                _adjacency[dst]['in'].append(src)
        self._adjacency = _adjacency
        
        
class NonlinearBoxModel(BoxModel):
    """ A representation of a non-linear box model to define fluxes
    between components in a system.

    Args:
        edges (list of edges): A list of weighted directed edges
            defining the connections and rate coefficients for the
            model. An edge has the form (src, dst, k(t, y)) where src
            is the source box, dst is the destination box and k(t, y)
            is a function that returns the rate coefficient for material
            flowing from src to dst when passed time and the unknown model
            state vector.
        nodes (list of labels, optional): A list of names for the nodes
            in the box model. By default, the names are inferred from
            the edges and given an arbitrary order. If the order of the
            nodes is important, nodes should list the boxes in the desired
            order.

    """

    def __init__(self, edges, nodes=None):
        super().__init__(edges, nodes)
        
    @property
    def coefficient_matrix(self):
        """The matrix K for the system y_dot = Ky + F."""
        def K(t, y):
            n = self.size
            _K = np.zeros((n, n))
            for node, props in self._adjacency.items():
                i = self._nodes.index(node)
                # Add negative coefficients for all the outgoing fluxes on diagonal
                _K[i, i] = -1 * np.sum([k[1](t, y) for k in props['out'].items()])
                # Add positive coefficients for incoming fluxes
                for src in props['in']:
                    j = self._nodes.index(src)
                    if node in self._adjacency[src]['out']:
                        k = self._adjacency[src]['out'][node]
                        _K[i, j] = k(t, y)
            return _K
        return K
        
    def _build_ode(self):
        # Nonlinear ode system
        K = self.coefficient_matrix
        f = lambda t, y: np.dot(K(t, y), y) + self.forcing_vector(t)
        self._ode = Ode(f)

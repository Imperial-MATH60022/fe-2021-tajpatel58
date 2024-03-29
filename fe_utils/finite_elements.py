# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
import scipy.special
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """

    if cell.dim == 1:
        # As 1D, initialise coordinates to 0 as y-coord == 0.
        coordiantes = np.array([[i/degree] for i in range(degree+1)])

        return coordiantes

    elif cell.dim == 2:
        # Using the set given in lectures, we can loop over to construct the points.
        # In particular, for some fixed i in {0,....,degree} the condition i+j<=degree
        # Tells us that 0<=i<=degree-j
        coordinates = np.array([[i/degree, j/degree] for j in range(degree+1) for i in range(0, degree-j+1)])

        return coordinates
    else:
        raise ValueError("We only accept cells in 1 or 2 dimensions")


#The following functions are used to evaluate x^power_x and x^power_x * y^power_y

def monomial_basis_1(point, power_x):
    return point[0]**power_x

# For our purposes we only want natural numbers to be powers, hence if we don't we shall return 0
# This is beneficial with regards to constructing the gradient vector.


def monomial_basis_2(point, power_x, power_y,):
    if power_x < 0 or power_y < 0:
        return 0
    else:
        return point[0] ** power_x * point[1] ** power_y


def derivative(point, power_x):
    if power_x == 0:
        return 0
    else:
        return power_x * monomial_basis_1(point, power_x-1)


def grad_vector(point, power_x, power_y):
    grad_1 = power_x * monomial_basis_2(point, power_x-1, power_y)
    grad_2 = power_y * monomial_basis_2(point, power_x, power_y-1)
    return [grad_1, grad_2]


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """


    """
    We firstly see that the number of rows of V is the number of points and 
    the number of columns of V is the dimension of P, which we can compute 
    as we've been given the formula. 
    
    """
    num_of_cols = int(scipy.special.comb(degree+cell.dim, cell.dim))

    """ 
    Seperate based on which dimension our reference is in and we 
    use the fact that a column of V corresponds to evaluating a 
    fixed basis function at all the different points in the reference.
    
    It's worth mentioning I will be using the monomoial basis.  
    """
    if grad:
        if cell.dim == 1:
            v = np.zeros((len(points), num_of_cols, 1))
            # Power denotes the power of x whose derivative we evaluate at.
            # Index starts at 1 because first column = 0.
            v = np.array([[[derivative(point, power)] for power in range(degree+1)] for point in points])
            return v
        else:
            v = np.zeros((len(points), num_of_cols, 2))
            col_num = 0
            # For a fixed order, we evaluate the basis functions of that order at points
            # e.g for order 2 we evaluate derivatives of x^2, xy, y^2 in that order.
            v = np.array([[grad_vector(point, order-power, power) for order in range(degree+1) for power in range(order+1)] for point in points])
            return v

    else:

        if cell.dim == 1:
            # Power denotes the power of x in our basis
            v = np.array([[monomial_basis_1(point, power) for power in range(degree+1)] for point in points])
            return v

        else:
            v = np.zeros((len(points), num_of_cols))
            col_num = 0
            # For a fixed order, we evaluate the basis functions of that order at points.
            v = np.array([[monomial_basis_2(point, order-power, power) for order in range(degree+1) for power in range(order+1)] for point in points])
            return v


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(self.cell, self.degree,self.nodes))

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.
        """
        if grad:
            # Use the fact that grad(phi)(X) = grad(V(X))*C ==> T = grad(V(X))*C
            # T_{i,j} is a linear combination of the gradient vectors in the ith row of V,
            # with coefficients given by the j^{th} column of C. (C is a 2D numpy array)
            v = vandermonde_matrix(self.cell, self.degree, points, grad=True)
            t = np.einsum("ijk,jl->ilk", v, self.basis_coefs)
            return t
        v = vandermonde_matrix(self.cell, self.degree, points)
        return np.matmul(v, self.basis_coefs)

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        return [fn(node) for node in self.nodes]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell, degree)
        entity_nodes = {}

        # Entity nodes in 1D is easy to compute, all nodes apart from end points lie on the edge, hence
        # associate to the edge. End points associated to vertices.
        if cell.dim == 1:
            entity_nodes = {0: {0: [0],
                                1: [len(nodes)-1]}}
            entity_nodes.update({1: {0: [j for j in range(1, len(nodes)-1)]}})

        elif cell.dim == 2:
            entity_nodes = {0: {0: [0],
                                1: [degree],
                                2: [len(nodes)-1]}}
            # list holding indexxes of nodes associated with vertices.
            nodes_verticies = [0, degree, len(nodes)-1]
            # lists holding the indeexes of nodes associated with corresponding edge.
            edge_0 = []
            edge_1 = []
            edge_2 = []
            # list holding indexxes of nodes within face of cell.
            face = []
            # The following code checks whether each nodes lies on an edge or within the face
            # and stores it within the appropriate list from above.
            for index, coord in enumerate(nodes):
                if index not in nodes_verticies:
                    if coord[1]+coord[0] == 1:
                        edge_0.append(index)
                    elif coord[0] == 0:
                        edge_1.append(index)
                    elif coord[1] == 0:
                        edge_2.append(index)
                    else:
                        face.append(index)
                else:
                    continue
            # topological_dim_j is a dictionary where keys is the numbering of the dimension j local mesh entities
            # and the values are lists of the indexes of nodes associated with that entity.

            toplogical_dim_1 = {0: edge_0,
                                1: edge_1,
                                2: edge_2}
            topological_dim_2 = {0: face}

            # Update entity nodes to include information of nodes associated with dimension 1 and 2.
            entity_nodes.update({1: toplogical_dim_1})
            entity_nodes.update({2: topological_dim_2})

        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes)

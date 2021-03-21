"""Solve a model poisson problem with Dirichlet boundary conditions
using the finite element method.

If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from __future__ import division
from fe_utils import *
import numpy as np
from numpy import sin, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs, f):
    """Assemble the finite element system for the Poisson problem given
    the function space in which to solve and the right hand side
    function."""


    # Create an appropriate (complete) quadrature rule.
    deg = fs.element.degree+2
    cell = fs.element.cell
    quad_rule = gauss_quadrature(cell, deg)

    # Tabulate the basis functions and their gradients at the quadrature points.
    basis_at_quad = fs.element.tabulate(quad_rule.points)
    basis_grad_at_quad = fs.element.tabulate(quad_rule.points, grad=True)

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    l = np.zeros(fs.node_count)

    # Now loop over all the cells and assemble A and l

    cell_node_map = fs.cell_nodes

    num_quad_point = len(quad_rule.points)

    num_of_cells = cell_node_map.shape[0]

    # List of global-nodes which lie on boundary as then we know which rows of A and l to construct differently.
    list_node_boundary = boundary_nodes(fs)

    for c in range(num_of_cells):
        # Note with our change of coordinates from cell c to reference cell, jacobian is constant in the cell.
        # Compute some essential Jacobian related terms.
        jacobian = fs.mesh.jacobian(c)
        det_j = np.abs(np.linalg.det(jacobian))
        J_inv_T = np.transpose(np.linalg.inv(jacobian))

        # Number of nodes in our ref cell.
        node_count_cell = fs.element.node_count

        # vector storing the values of f at the nodes of cell c
        f_val_in_cell_c = [f.values[cell_node_map[c, k]] for k in range(node_count_cell)]

        # vector where q^{th} component stores values of f at the q^th quadrature point.
        integral_f_cell_c = [np.dot(f_val_in_cell_c, basis_at_quad[q, :]) for q in range(num_quad_point)]

        # Constructing the matrix A and l.
        for i in range(node_count_cell):
            # value of element in vector l.
            integral_l = 0
            # row_index for A and for l.
            row_index = cell_node_map[c, i]
            if row_index in list_node_boundary:
                A[np.ix_(np.array([row_index]), np.array([row_index]))] = 1
            else:
                for q in range(num_quad_point):
                    integral_l += basis_at_quad[q, i] * integral_f_cell_c[q] * quad_rule.weights[q]*det_j
                # update the correct element of l.
                l[cell_node_map[c, i]] += integral_l
                for j in range(node_count_cell):
                    integral = 0
                    for q in range(num_quad_point):
                        # Compute different components of the integral at different quadrature points.
                        grad_term = np.dot((J_inv_T  @  basis_grad_at_quad[q, i]), (J_inv_T @ basis_grad_at_quad[q, j]))
                        integral += grad_term * quad_rule.weights[q] * det_j
                    # column index for the entry of A we may change.
                    col_index = cell_node_map[c, j]
                    # Only edit the matrix A if the integral is non-zero.
                    if integral != 0:
                        #A[np.ix_(np.array([row_index]), np.array([col_index]))] += np.array([integral])
                        A[np.ix_([row_index], [col_index])] += np.array([integral])
                    else:
                        continue

    return A, l



def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def solve_poisson(degree, resolution, analytic=False, return_error=False):
    """Solve a model Poisson problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: sin(4*pi*x[0])*x[1]**2*(1.-x[1])**2)

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: (16*pi**2*(x[1] - 1)**2*x[1]**2 - 2*(x[1] - 1)**2 -
                             8*(x[1] - 1)*x[1] - 2*x[1]**2) * sin(4*pi*x[0]))

    # Assemble the finite element system.
    A, l = assemble(fs, f)

    # Create the function to hold the solution.
    u = Function(fs)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error

if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve a Poisson problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_poisson(degree, resolution, analytic, plot_error)

    u.plot()

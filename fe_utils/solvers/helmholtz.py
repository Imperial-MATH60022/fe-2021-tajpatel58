"""Solve a model helmholtz problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
from numpy import cos, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs, f):
    """Assemble the finite element system for the Helmholtz problem given
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
    for c in range(num_of_cells):
        # Note with our change of coordinates from cell c to reference cell, jacobian is constant in the cell.
        jacobian = fs.mesh.jacobian(c)
        det_j = np.abs(np.linalg.det(jacobian))

        # Number of nodes in our ref cell.
        node_count_cell = fs.element.node_count

        # vector storing the values of f at the nodes of cell c
        f_val_in_cell_c = [f.values[cell_node_map[c, k]] for k in range(node_count_cell)]

        # vector where q^{th} component stores values of f at the q^th quadrature point.
        integral_f_cell_c = [np.dot(f_val_in_cell_c, basis_at_quad[q, :]) for q in range(num_quad_point)]

        # The following adds the integral to the correct component of l.
        for i in range(node_count_cell):
            integral = 0
            for q in range(num_quad_point):
                integral += basis_at_quad[q, i] * integral_f_cell_c[q] * quad_rule.weights[q]*det_j
            l[cell_node_map[c, i]] += integral


        ########## NOW WE CONSTRUCT THE RIGHT HAND SIDE ###########

        # construct the inverse of the jacobian and the transpose
        J_inv_T = np.transpose(np.linalg.inv(jacobian))
        J_inv = np.linalg.inv(jacobian)
        for i in range(node_count_cell):
            for j in range(node_count_cell):
                integral = 0
                integral_1 = 0
                for q in range(num_quad_point):
                    grad_term = np.dot((J_inv_T  @  basis_grad_at_quad[q, i]), (J_inv_T @ basis_grad_at_quad[q, j]))
                    product_term = basis_at_quad[q, i] * basis_at_quad[q, j]
                    integral += (grad_term + product_term) * quad_rule.weights[q] * det_j
                    """
                    for a in range(2):
                        for b in range(2):
                            for g in range(2):
                                integral_1 += (J_inv[b, a] * basis_grad_at_quad[q, i, b] * J_inv[g, a] * basis_grad_at_quad[q, j, g] + basis_at_quad[q, i] * basis_at_quad[q, j]) * det_j * quad_rule.weights[q]
                    """
                row_index = cell_node_map[c, i]
                col_index = cell_node_map[c, j]
                if integral != 0:
                    A[np.ix_(np.array([row_index]), np.array([col_index]))] += np.array([integral])
                else:
                    continue
    return A, l


def solve_helmholtz(degree, resolution, analytic=False, return_error=False):
    """Solve a model Helmholtz problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: cos(4*pi*x[0])*x[1]**2*(1.-x[1])**2)

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: ((16*pi**2 + 1)*(x[1] - 1)**2*x[1]**2 - 12*x[1]**2 + 12*x[1] - 2) *
                  cos(4*pi*x[0]))

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

solve_helmholtz(1,1)

if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve a Helmholtz problem on the unit square.""")
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

    u, error = solve_helmholtz(degree, resolution, analytic, plot_error)

    u.plot()

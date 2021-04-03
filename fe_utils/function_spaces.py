import numpy as np
from . import ReferenceTriangle, ReferenceInterval
from .finite_elements import LagrangeElement, lagrange_points
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from mesh import unit_square, unit_interval
import quadrature


class FunctionSpace(object):
    def __init__(self, mesh, element):
        """A finite element space.

        :param mesh: The :class:`~.mesh.Mesh` on which this space is built.
        :param element: The :class:`~.finite_elements.FiniteElement` of this space.

        Most of the implementation of this class is left as an :ref:`exercise
        <ex-function-space>`.
        """

        #: The :class:`~.mesh.Mesh` on which this space is built.
        self.mesh = mesh
        #: The :class:`~.finite_elements.FiniteElement` of this space.
        self.element = element

        # Implement global numbering in order to produce the global
        # cell node list for this space.
        #: The global cell node list. This is a two-dimensional array in
        #: which each row lists the global nodes incident to the corresponding
        #: cell. The implementation of this member is left as an
        #: :ref:`exercise <ex-function-space>`

        # The following vector helps us yield the first term of our global numbering for an entity.
        global_num_vec = [0] * (mesh.dim+1)
        for ind in range(1, mesh.dim+1):
            global_num_vec[ind] = mesh.entity_counts[ind-1] * element.nodes_per_entity[ind-1] + global_num_vec[ind-1]

        # Number of rows of cell_nodes is the number of cells.
        nrows = mesh.entity_counts[-1]
        # Number of columns is number of nodes in each cell.
        ncols = element.node_count
        # Initalise the matrix.
        cell_nodes = np.zeros((nrows,ncols), dtype=int)
        # Establish the rows as we go along.
        for row_index in range(nrows):
            for delta in range(mesh.dim):
                #adj_ents_num denotes the indexxes of the adjacent delta dimensional entities.
                adj_ents_num = mesh.adjacency(mesh.dim, delta)[row_index]
                # Below corresponds to picking the entity (delta, adj_ents_num[e])
                for e, ent in enumerate(adj_ents_num):
                    if element.entity_nodes[delta][e]:
                        # global numbering value corresponding to this entity.
                        g_d_i = global_num_vec[delta] + ent*element.nodes_per_entity[delta]
                        # After first node value assigned, we need to increment by one for a value of the next node
                        # on this entity.
                        for index in element.entity_nodes[delta][e]:
                            cell_nodes[row_index, index] = g_d_i
                            g_d_i += 1
                    else:
                        continue
            # Need to account for the adjacency when dim1 == dim2, following stores node values within the cell:

            # The following denotes the value of the first node inside the cell which has index = row_index
            first_node_in_cell = global_num_vec[-1] + row_index * element.nodes_per_entity[-1]
            # We loop over the indices that store the value of the nodes contained in the cell.
            for index in element.entity_nodes[mesh.dim][0]:
                cell_nodes[row_index, index] = first_node_in_cell
                first_node_in_cell += 1

        self.cell_nodes = cell_nodes

        #: The total number of nodes in the function space.
        self.node_count = np.dot(element.nodes_per_entity, mesh.entity_counts)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.mesh,
                               self.element)


class Function(object):
    def __init__(self, function_space, name=None):
        """A function in a finite element space. The main role of this object
        is to store the basis function coefficients associated with the nodes
        of the underlying function space.

        :param function_space: The :class:`FunctionSpace` in which
            this :class:`Function` lives.
        :param name: An optional label for this :class:`Function`
            which will be used in output and is useful for debugging.
        """

        #: The :class:`FunctionSpace` in which this :class:`Function` lives.
        self.function_space = function_space

        #: The (optional) name of this :class:`Function`
        self.name = name

        #: The basis function coefficient values for this :class:`Function`
        self.values = np.zeros(function_space.node_count)

    def interpolate(self, fn):
        """Interpolate a given Python function onto this finite element
        :class:`Function`.

        :param fn: A function ``fn(X)`` which takes a coordinate
          vector and returns a scalar value.

        """

        fs = self.function_space

        # Create a map from the vertices to the element nodes on the
        # reference cell.
        cg1 = LagrangeElement(fs.element.cell, 1)
        coord_map = cg1.tabulate(fs.element.nodes)
        cg1fs = FunctionSpace(fs.mesh, cg1)

        for c in range(fs.mesh.entity_counts[-1]):
            # Interpolate the coordinates to the cell nodes.
            vertex_coords = fs.mesh.vertex_coords[cg1fs.cell_nodes[c, :], :]
            node_coords = np.dot(coord_map, vertex_coords)

            self.values[fs.cell_nodes[c, :]] = [fn(x) for x in node_coords]

    def plot(self, subdivisions=None):
        """Plot the value of this :class:`Function`. This is quite a low
        performance plotting routine so it will perform poorly on
        larger meshes, but it has the advantage of supporting higher
        order function spaces than many widely available libraries.

        :param subdivisions: The number of points in each direction to
          use in representing each element. The default is
          :math:`2d+1` where :math:`d` is the degree of the
          :class:`FunctionSpace`. Higher values produce prettier plots
          which render more slowly!

        """

        fs = self.function_space

        d = subdivisions or (2 * (fs.element.degree + 1) if fs.element.degree > 1 else 2)

        if fs.element.cell is ReferenceInterval:
            fig = plt.figure()
            fig.add_subplot(111)
            # Interpolation rule for element values.
            local_coords = lagrange_points(fs.element.cell, d)

        elif fs.element.cell is ReferenceTriangle:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            local_coords, triangles = self._lagrange_triangles(d)

        else:
            raise ValueError("Unknown reference cell: %s" % fs.element.cell)

        function_map = fs.element.tabulate(local_coords)

        # Interpolation rule for coordinates.
        cg1 = LagrangeElement(fs.element.cell, 1)
        coord_map = cg1.tabulate(local_coords)
        cg1fs = FunctionSpace(fs.mesh, cg1)

        for c in range(fs.mesh.entity_counts[-1]):
            vertex_coords = fs.mesh.vertex_coords[cg1fs.cell_nodes[c, :], :]
            x = np.dot(coord_map, vertex_coords)

            local_function_coefs = self.values[fs.cell_nodes[c, :]]
            v = np.dot(function_map, local_function_coefs)

            if fs.element.cell is ReferenceInterval:

                plt.plot(x[:, 0], v, 'k')

            else:
                ax.plot_trisurf(Triangulation(x[:, 0], x[:, 1], triangles),
                                v, linewidth=0)

        plt.show()

    @staticmethod
    def _lagrange_triangles(degree):
        # Triangles linking the Lagrange points.

        return (np.array([[i / degree, j / degree]
                          for j in range(degree + 1)
                          for i in range(degree + 1 - j)]),
                np.array(
                    # Up triangles
                    [np.add(np.sum(range(degree + 2 - j, degree + 2)),
                            (i, i + 1, i + degree + 1 - j))
                     for j in range(degree)
                     for i in range(degree - j)]
                    # Down triangles.
                    + [np.add(np.sum(range(degree + 2 - j, degree + 2)),
                              (i+1, i + degree + 1 - j + 1, i + degree + 1 - j))
                       for j in range(degree - 1)
                       for i in range(degree - 1 - j)]))

    def integrate(self):
        """Integrate this :class:`Function` over the domain.

        :result: The integral (a scalar).
        """

        # If our polynomial is of degree p, then we need a quadrature rule of degree of precision >= p.
        cell = self.function_space.element.cell
        deg = self.function_space.element.degree
        # Our Quadrature rule.
        quad_rule = quadrature.gauss_quadrature(cell, deg)

        # Now we need to evaluate our nodal basis at the quadrature points:
        basis_at_quad = self.function_space.element.tabulate(quad_rule.points)
        cell_nodes_map = self.function_space.cell_nodes
        num_of_cells = cell_nodes_map.shape[0]

        integral = 0
        # Loop over the cells and sum over them.
        for cell in range(num_of_cells):
            # Jacobian is fixed per cell.
            J = np.absolute(np.linalg.det(self.function_space.mesh.jacobian(cell)))
            F_vec = np.take(self.values, cell_nodes_map[cell, :])
            integral += np.einsum('i,q,qi->',F_vec, quad_rule.weights,basis_at_quad)*J
        return integral





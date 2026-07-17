"""MVP demo for the pyGeotech finite-element seepage engine.

Confined flow through a heterogeneous foundation soil that contains a
low-permeability clay lens; the equipotentials bend around the lens and
the flow accelerates through the more permeable soil.

Run from the repository root:

    python examples/demo_fem_seepage.py
"""

import numpy as np

from pygeotech.fem import SeepageFEM, rectangular_mesh
from pygeotech.fem.plotting import plot_seepage


def main() -> None:
    width, height = 20.0, 8.0
    nx, ny = 80, 32
    mesh = rectangular_mesh(width, height, nx, ny)

    # Base soil k = 1e-4 m/s; a central clay lens is 100x less permeable.
    k = np.full(mesh.n_elements, 1e-4)
    centroids = mesh.element_centroids()
    in_lens = (
        (centroids[:, 0] > 7.0) & (centroids[:, 0] < 13.0)
        & (centroids[:, 1] > 2.5) & (centroids[:, 1] < 5.5)
    )
    k[in_lens] = 1e-6

    fem = SeepageFEM(mesh, conductivity=k)
    fem.set_head(mesh.nodes_where("x", 0.0), 8.0)     # upstream head
    fem.set_head(mesh.nodes_where("x", width), 2.0)   # downstream head
    fem.solve()

    q = fem.boundary_flow(mesh.nodes_where("x", 0.0))
    print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"Seepage flow rate Q = {q:.3e} m^3/s per metre run")
    print(f"Max Darcy velocity  = {np.linalg.norm(fem.velocities(), axis=1).max():.3e} m/s")

    plot_seepage(fem, n_equipotentials=14,
                 save_path="examples/output/fem_seepage.png")
    plot_seepage(fem, n_equipotentials=14,
                 save_path="examples/output/fem_seepage.pdf")
    print("Figure -> examples/output/fem_seepage.{png,pdf}")


if __name__ == "__main__":
    main()

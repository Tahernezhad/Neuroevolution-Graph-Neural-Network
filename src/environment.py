import numpy as np


class Environment():
    def __init__(self, reactions, loads):
        self.reactions = reactions
        self.loads = loads

    def update_physical_state(self, nodes, edges, cs_areas, physical_state, material):
        try:
            mem_lens = self.cal_mem_lens(nodes, edges)
            physical_state.volumes = self._cal_member_volume(mem_lens, cs_areas)
            equilibrium_mat = self._get_equilibrium_mat(nodes, edges, self.reactions, dim=2)

            # Use np.linalg.solve for stability and speed instead of inv()
            tension_coeffs = np.linalg.solve(equilibrium_mat, -self.loads.reshape(-1, ))

            physical_state.forces = self._cal_member_forces(tension_coeffs, self.reactions, mem_lens)
            physical_state.stresses = self._cal_member_stresses(physical_state.forces, cs_areas)
            physical_state.strain_energies = self._cal_member_strain_energies(physical_state.stresses,
                                                                              physical_state.volumes,
                                                                              material.young_mods)
        except np.linalg.LinAlgError:
            # If the truss is unstable, assign a terrible fitness.
            num_edges = edges.shape[0]
            physical_state.volumes = np.full((num_edges,), 1e9)
            physical_state.strain_energies = np.full((num_edges,), 1e9)
            physical_state.forces = np.zeros((num_edges,))
            physical_state.stresses = np.zeros((num_edges,))

        return physical_state

    def cal_mem_lens(self, nodes: np.ndarray, members: np.ndarray) -> np.ndarray:
        """Calculate the length of each member in truss.

        - Finds the Euclidean distance between the nodes of each member in the truss.

        :param nodes: 2D array of coordinates for each node.
        :param members: 2D array of node indices for each member/edge.
        :return: 1D array of lengths for each member.
        """
        vecs = nodes[members[:, 1]] - nodes[members[:, 0]]
        mems_lens = np.sqrt(np.sum(np.power(vecs, 2), axis=1))
        return mems_lens

    def _get_equilibrium_mat(self, 
                            nodes: np.ndarray,
                            edges: np.ndarray,
                            reactions: np.ndarray,
                            dim: int = 2) -> np.ndarray:
        """Creates the equilibrium equation matrix used for solving the forces in the truss.

        - Loops over each node in the truss.
        - For each edge associated with a given node, the vector components of the edge from the given node is found in
        the x, y or z directions (depending on dimensionality of truss) and added to the equilibrium equation matrix.
        - Then the reactions on the system are also added to the equilibrium matrix.

        :param nodes: 2D array of coordinates for each node.
        :param edges: 2D array of node indices for each edge.
        :param reactions: 1D array of reactions at each node in the x & y or x, y & z dimension.
        :param dim: Dimensionality of the truss (i.e. 2D==2 or 3D==3)
        :return: 2D array of representing the equilibrium equation matrix.
        """
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        num_nodes = np.shape(nodes)[0]
        equilibrium_mat = np.zeros((num_nodes * dim, num_nodes * dim), dtype=np.float32)

        for node in range(num_nodes):
            row = node * dim

            for edge_idx, edge in enumerate(edges):
                if node in edge:
                    node_idx = np.argwhere(edge == node)

                    if node_idx == 0:
                        equilibrium_mat[row, edge_idx] = nodes[edge[1], 0] - nodes[edge[0], 0]
                        equilibrium_mat[row + 1, edge_idx] = nodes[edge[1], 1] - nodes[edge[0], 1]

                        if dim > 2:
                            equilibrium_mat[row + 2, edge_idx] = nodes[edge[1], 2] - nodes[edge[0], 2]

                    else:
                        equilibrium_mat[row, edge_idx] = nodes[edge[0], 0] - nodes[edge[1], 0]
                        equilibrium_mat[row + 1, edge_idx] = nodes[edge[0], 1] - nodes[edge[1], 1]

                        if dim > 2:
                            equilibrium_mat[row + 2, edge_idx] = nodes[edge[0], 2] - nodes[edge[1], 2]

        reactions = reactions.reshape(-1, )
        row_idxs = np.argwhere(reactions != 0)
        num_edges = np.shape(edges)[0]
        num_reactions = np.count_nonzero(reactions)
        col_idxs = np.arange(num_edges, num_edges + num_reactions).reshape((-1, 1))
        equilibrium_mat[row_idxs, col_idxs] = reactions[row_idxs]

        return equilibrium_mat

    def _cal_tension_coeffs(self, equilibrium_mat: np.ndarray, loads: np.ndarray) -> np.ndarray:
        """Calculates the tension co-efficients and reaction forces for the truss.

        - Find the inverse of the equilibrium matrix and matrix multiple with the negative load vector
        to calculate the tension co-efficients for each member and reaction forces.

        :param equilibrium_mat: 2D array describing the equilibrium equations of system.
        :param loads: 1D array describing the loads at each node in the x & y or x, y & z directions.
        :return: 1D array of tension co-efficients for each member/edge.
        """
        tension_coeffs = np.dot(np.linalg.inv(equilibrium_mat), -loads.reshape(-1,))
        return tension_coeffs

    def _cal_member_forces(self, tension_coeffs: np.ndarray, reactions: np.ndarray, mem_lens: np.ndarray) -> np.ndarray:
        """Calculate forces in each member of the truss.

        - The tension coeffs array will contain the forces for each reaction and will not be needed.
        - The forces for each member is found by multipling the tension co-efficients by the edge lengths.

        :param tension_coeffs: 1D array of tension co-efficients for each member/edge.
        :param reactions: 1D array of reactions at each node in the x & y or x, y & z dimension.
        :param mem_lens: 1D array of edge lengths for each member/edge.
        :return: 1D array of forces for each member/edge.
        """
        num_reactions = np.count_nonzero(reactions)
        mem_forces = tension_coeffs[:-num_reactions] * mem_lens
        return mem_forces

    def _cal_member_volume(self, mem_lens: np.ndarray, cross_areas: np.ndarray) -> np.ndarray:
        """Calculate volume for each member in truss.

        :param mem_lens: 1D array of lengths for each member/edge.
        :param cross_areas: 1D array of cross-sectional areas for each member/edge.
        :return: 1D array of volumes for each member/edge.
        """
        mem_vols = mem_lens * cross_areas
        return mem_vols

    def _cal_total_volume(self, mem_vols: np.ndarray) -> np.ndarray:
        """Calculate total volume of system.

        :param mem_vols: 1D array of volumes for each member/edge.
        :return: 1D array of total volume of system.
        """
        total_vol = np.sum(mem_vols)
        return total_vol

    def _cal_member_stresses(self, mem_forces: np.ndarray, cross_areas: np.ndarray) -> np.ndarray:
        """Calculates the stresses on each member in the truss.

        :param mem_forces: 1D array of forces for each member/edge.
        :param cross_areas: 1D array of cross-sectional areas for each member/edge.
        :return: 1D array of stresses for each member/edge.
        """
        mem_stresses = mem_forces / cross_areas
        return mem_stresses

    def _cal_member_strain_energies(self, mem_stresses: np.ndarray, mem_vols: np.ndarray, young_mods: np.ndarray) -> np.ndarray:
        """Calculate the strain energies for each member in truss.

        :param mem_stresses: 1D array of stresses for each member/edge.
        :param mem_vols: 1D array of stresses for each member/edge.
        :param young_mods: 1D array of Young's Modulus for each member/edge.
        :return: 1D array of strain energies for each member/edge.
        """
        mem_strains = 0.5 * np.power(mem_stresses, 2) * mem_vols / young_mods
        return mem_strains

    def _cal_total_strain_energy(self, mem_strain_energies: np.ndarray) -> np.ndarray:
        """Calculates the total strain energy for the system.

        :param mem_strains: 1D array of strain energies for each member/edge.
        :return: 1D array containing the total strain energy for the system.
        """
        total_strain_energy = np.sum(mem_strain_energies)
        return total_strain_energy

    def _cal_member_strains(self, mem_stresses, mem_mods):
        return mem_stresses / mem_mods

    def _cal_joint_displacements(self, nodes, mem_lens, reactions, equilibrium_mat, mem_strains, dim=2):
        mem_deforms = mem_strains * mem_lens
        num_nodes = np.shape(nodes)[0]
        num_node_dims = num_nodes * dim
        joint_displacements = []
        joint_disp = []
        cnt = 0

        for node in range(num_node_dims):
            virtual_load = np.zeros(num_node_dims)
            virtual_load[node] = 1
            virtual_t_coeffs = self._cal_tension_coeffs(equilibrium_mat, virtual_load)
            virtual_forces = self._cal_member_forces(virtual_t_coeffs, reactions, mem_lens)
            joint_disp.append(np.sum(virtual_forces * mem_deforms))
            cnt += 1

            if cnt > 2:
                joint_displacements.append(joint_disp)
                joint_disp = []
                cnt = 0

        return np.array(joint_displacements)

    def _cal_mem_buckling(self, mem_stresses: np.ndarray, mem_lens: np.ndarray,
                        mem_mods: np.ndarray, mem_gyration_rads: np.ndarray) -> np.ndarray:
        """Calculate which members buckle under a given stress.

        :param mem_stresses: Stresses for each edge in artefact.
        :param mem_lens: Lengths for each edge in artefact.
        :param mem_mods: Young's moduli for each edge in artefact.
        :param mem_gyration_rads: Gyration radii for each edge in artefact.
        :return: Boolean array of which edges buckled.
        """
        mem_crit_stresses = np.pi ** 2 * mem_mods / np.power(mem_lens / mem_gyration_rads, 2)
        mem_buckling = np.where((mem_stresses < 0) & (np.abs(mem_stresses) >= np.abs(mem_crit_stresses)), 1.0, 0.0)
        return mem_buckling
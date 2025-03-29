# ==============================================================================================================================
#  SebMC, a budget Monte Carlo neutron transport code
# ==============================================================================================================================

from __future__ import annotations
import numpy as np
from typing import List, Optional
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import multiprocessing
import random
import time

DELTA_TRACKING = 0
random.seed(1)

c = 3e10                # Speed of light cm/s
Mn = 938/c**2           # Mass of neutron
E0 = 5000               # Initial neutron energy eV
v0 = np.sqrt(2*E0/Mn)   # Initial neutron speed cm/s
nubar = 2.42            # Number of neutrons per fission

class Vector3D(np.ndarray):
    def __new__(cls, x, y, z):
        return np.array([x, y, z], dtype=np.float64).view(cls)

class Nuclide:
    # Class to hold name, atomic density, and cross sections for a single nuclide
    def __init__(self, name:str, nuclide_number:np.float64, a_density:np.float64):
        self.name = name
        self.A = nuclide_number
        self.a_density = a_density        
        self.xs_fission_0 = 0
        self.xs_absorbtion_0 = 0
        self.E0 = 0
        self.Gamma = 0
        self.xs_scatter_0 = 0

class Material:
    # Class to hold many nuclides
    def __init__(self, nuclides:list[Nuclide], temperature:np.float64):
        self.nuclides = nuclides
        self.temperature = temperature

    def resAbs(self, E, E0, Gamma, T):
        kB = 8.617333262e-5  # Boltzmann constant in eV/K
        
        # Breit-Wigner function
        def breit_wigner(Eprime):
            return Gamma**2 / ((Eprime - E0)**2 + Gamma**2)
        
        # Doppler broadening Gaussian kernel
        def doppler_kernel(Eprime, E):
            return (1 / np.sqrt(4 * np.pi * kB * T * E0)) * \
                np.exp(-((Eprime - E)**2) / (4 * kB * T * E0))
        
        # Integrand function
        def integrand(Eprime):
            return breit_wigner(Eprime) * doppler_kernel(Eprime, E)
        
        # Integration limits
        Emin = E0 - 10 * Gamma
        Emax = E0 + 10 * Gamma
        
        # Perform numerical integration
        fD, _ = quad(integrand, Emin, Emax)
        
        return fD
    
    def get_all_Macro_XS(self, energy:np.float64):
        XS_dict = {}
        for i, nuclide in enumerate(self.nuclides):
            if nuclide.xs_fission_0 != 0: XS_dict["f"+str(i)] = nuclide.a_density * 1e-24 * nuclide.xs_fission_0 / np.sqrt(energy)
            if nuclide.xs_absorbtion_0 != 0: XS_dict["a"+str(i)] = nuclide.a_density * 1e-24 * nuclide.xs_absorbtion_0 * self.resAbs(energy, nuclide.E0, nuclide.Gamma, self.temperature)
            if nuclide.xs_scatter_0 != 0: XS_dict["s"+str(i)] = nuclide.a_density * 1e-24 * nuclide.xs_scatter_0
        # print (XS_dict)
        return XS_dict

    def get_total_Macro_XS(self, energy:np.float64):
        XS_dict = self.get_all_Macro_XS(energy)
        return sum(XS_dict.values())

class Cell:
    # spherical geometry object
    def __init__(self, cell_id: int, boundary_type: int, importance: int, material: Material, center: Vector3D, radius: np.float64):
        self.id = cell_id
        self.boundary_type = boundary_type  # 0: transparent, 1: kill, 2: reflection
        self.importance = importance
        self.material = material
        self.center = center
        self.radius = radius
        self.children: List[Cell] = []  # Tree structure: holds contained spheres

    def contains(self, other:Cell) -> bool:
        """Check if this sphere completely contains another sphere."""
        distance = np.linalg.norm(self.center - other.center)
        return distance + other.radius < self.radius

    def add_child(self, child:Cell):
        """Add a sphere as a child node, ensuring it is properly nested."""
        for existing_child in self.children:
            if existing_child.contains(child):  
                existing_child.add_child(child)  
                return 
        self.children.append(child)  # If no existing child contains it, add it here

    def line_sphere_intersection(self, pos: Vector3D, vel: Vector3D):
        """Computes the closest intersection distance of a line with a sphere centered at self.center."""
        # pos = np.array(pos)  # Convert to NumPy arrays
        # vel = np.array(vel)
        # center = np.array(self.center)
        
        # Shift pos to be relative to the sphere's center
        r = pos - self.center  # New origin-centered position
        vel = vel / np.linalg.norm(vel)
        
        # Quadratic coefficients for intersection calculation
        a = np.dot(vel, vel)  # Should be 1 if vel is normalized
        b = 2 * np.dot(r, vel)
        c = np.dot(r, r) - self.radius**2  # Distance from sphere surface

        # Compute discriminant
        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0:
            return None  # No intersection
        elif discriminant == 0:
            t = -b / (2 * a)
            return t if t >= 0 else None  # Tangent intersection (only if it's in front)
        else:
            # Two possible intersection points
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)

            # Sort intersections to get the closest one
            t_min, t_max = sorted([t1, t2])

            # If pos is inside the sphere, return the exit point (t_max)
            if c < 0:
                return t_max

            # If both intersections are behind the ray, return None
            if t_max < 0:
                return None

            # Return the closest valid intersection
            return t_min if t_min >= 0 else t_max

class Geometry:
    def __init__(self, cells: List[Cell]):
        if not self.intersection_check(cells):
            if not self.id_check(cells):

                self.cells = sorted(cells, key=lambda cell: cell.radius, reverse=True)
                self.root: Optional[Cell] = None
                self.build_tree(cells)

            else: raise AttributeError("Cells must have unique IDs!")
        else: raise AttributeError("Cells must not overlap!")

    def build_tree(self, cells: List[Cell]):
        if not cells:
            return
        
        # Sort cells by radius in descending order (largest first)
        cells.sort(key=lambda c: c.radius, reverse=True)
        self.root = cells[0]  # The largest sphere is the root
        
        # Insert each sphere into the tree
        for cell in cells[1:]:
            self.root.add_child(cell)

    def find_cell(self, cell_id: int, node: Optional[Cell] = None) -> Optional[Cell]:
        """Recursively searches for a cell with the given ID."""
        if node is None:
            node = self.root
        if node is None:
            return None
        if node.id == cell_id:
            return node
        for child in node.children:
            result = self.find_cell(cell_id, child)
            if result:
                return result
        return None

    def print_tree(self, node: Optional[Cell] = None, level=0):
        """Recursively print the tree structure."""
        if node is None:
            node = self.root
        if node is None:
            return
        print("  " * level + f"Sphere {node.id}, Radius {node.radius}")
        for child in node.children:
            self.print_tree(child, level + 1)

    def intersection_check(self, cells):
        """Checks if any pair of spheres in the list of cells intersect.
        Allows nesting, but ensures no two spheres touch each other's surface."""
        for i, cell1 in enumerate(cells):
            for j, cell2 in enumerate(cells):
                if i < j:  # Avoid checking the same pair twice
                    # Calculate the distance between the centers of the two spheres
                    distance = np.linalg.norm(cell1.center - cell2.center)
                    # Check if the spheres touch or overlap
                    if distance < (cell1.radius + cell2.radius) and distance > np.abs(cell1.radius - cell2.radius):
                        return True
        return False

    def id_check(self, cells:np.ndarray[Cell]):
        """ensure no 2 cells have the same id!"""
        seen = set()

        for cell in cells:
            if cell.id in seen:
                return True
            seen.add(cell.id)

        return False
    
    def find_containing_cell(self, pos: Vector3D, node: Optional[Cell] = None) -> Optional[Cell]:
        """
        Finds the deepest cell that contains the given position.
        Starts from root and recursively checks child spheres.
        """
        if node is None:
            node = self.root
        if node is None or np.linalg.norm(pos - np.array(node.center)) > node.radius:
            return None  # Outside this sphere

        # Recursively check child cells
        for child in node.children:
            containing_cell = self.find_containing_cell(pos, child)
            if containing_cell:
                return containing_cell  # Return the deepest cell containing pos
        
        return node  # If no deeper cell is found, return this one
    
    def find_containing_cell_bruteforce(self, pos:Vector3D) -> Cell:
        """Determines which sphere (Cell) contains a given position in 3D space."""
        
        # Find all cells that contain the point
        containing_cells = [cell for cell in self.cells if np.linalg.norm(pos - cell.center) < cell.radius]
        
        if not containing_cells:
            return None  # The point is not inside any cell
        
        # Return the smallest enclosing cell (by radius)
        return min(containing_cells, key=lambda cell: cell.radius)

    def distance_to_next_boundary(self, pos: Vector3D, vel: Vector3D) -> Optional[float]:
        """
        Calculates the distance to the next boundary.
        - Finds the current cell containing `pos`.
        - Computes the intersection with its boundary and child boundaries.
        - Returns the minimum positive intersection distance.
        """
        vel = vel / np.linalg.norm(vel)
        # Step 1: Find the cell containing pos
        current_cell = self.find_containing_cell(pos)
        if not current_cell:
            return None  # Outside all cells

        # Step 2: Get intersections with the current cell and its children
        distances = []

        # Check intersection with current cell
        dist = current_cell.line_sphere_intersection(pos, vel)
        if dist is not None:
            distances.append(dist)

        # Check intersections with child cells
        for child in current_cell.children:
            child_dist = child.line_sphere_intersection(pos, vel)
            if child_dist is not None:
                distances.append(child_dist)

        # Step 3: Return the minimum valid distance
        return min(distances) if distances else None
        # min_distance = min(distances)
        # if min_distance < 1e-14: return 0
        # else: return min_distance 

    def plot_slice(self, plane='xy', value=0.0):
        """Plots a 2D slice of the geometry in the specified plane."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')

        # Plane mappings
        plane_axes = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        if plane not in plane_axes:
            raise ValueError("Invalid plane. Choose from 'xy', 'xz', or 'yz'.")

        idx1, idx2 = plane_axes[plane]
        min_vals, max_vals = [np.inf, np.inf], [-np.inf, -np.inf]

        # Sort the cells by radius in descending order
        sorted_cells = sorted(self.cells, key=lambda cell: cell.radius, reverse=True)

        for cell in sorted_cells:
            slice_dist = np.abs(cell.center[3 - idx1 - idx2] - value)
            if slice_dist < cell.radius:
                slice_radius = np.sqrt(cell.radius**2 - slice_dist**2)

                # Assign a random color for each region
                color = plt.cm.viridis(random.random())  # You can change colormap here
                circle = plt.Circle((cell.center[idx1], cell.center[idx2]), slice_radius, 
                                    facecolor=color, edgecolor='k', linewidth=0.5)
                ax.add_patch(circle)

                # Track min/max values for auto-scaling
                min_vals[0] = min(min_vals[0], cell.center[idx1] - slice_radius)
                min_vals[1] = min(min_vals[1], cell.center[idx2] - slice_radius)
                max_vals[0] = max(max_vals[0], cell.center[idx1] + slice_radius)
                max_vals[1] = max(max_vals[1], cell.center[idx2] + slice_radius)

        # Set axis limits based on the detected min/max extents
        if np.isfinite(min_vals[0]) and np.isfinite(max_vals[0]):
            ax.set_xlim(min_vals[0], max_vals[0])
            ax.set_ylim(min_vals[1], max_vals[1])
        else:
            ax.set_xlim(-1, 1)  # Default limits if no spheres are found
            ax.set_ylim(-1, 1)

        ax.set_xlabel(['X', 'Y', 'Z'][idx1])
        ax.set_ylabel(['X', 'Y', 'Z'][idx2])
        ax.set_title(f"{plane.upper()} plane slice at {['X', 'Y', 'Z'][3 - idx1 - idx2]} = {value}")
        plt.grid(True)
        # plt.show()
        plt.savefig(f"{plane.upper()}_slice_at_{['X', 'Y', 'Z'][3 - idx1 - idx2]}={value}.png")
        plt.close()

class Track:
    def __init__(self, geometry:Geometry, source_pos:Vector3D, source_energy:np.float64):
        self.geometry = geometry
        self.source_pos = source_pos
        self.source_energy = source_energy
        self.neutron_weight = 1
        self.history = [] # list where each position has: [collision type, pos, dir, energy] 
        # collision types: "i": initial, "b":boundary, "k": kill, "r":reflect, "s": scatter, "a":absorbtion, "f": fission, "v": virtual

    def get_tracklength(self, cell:Cell, energy:np.float64) -> np.float64:
        # random path length of travel
        return - np.log(random.random()) / cell.material.get_total_Macro_XS(energy)

    def get_isotropic_dir(self) -> Vector3D:
        # random direction of travel
        phi = 2 * np.pi * random.random()
        theta = np.arccos(2 * random.random() - 1)
        return Vector3D(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))

    def get_reaction(self, cell:Cell, energy:np.float64) -> list[str,np.float64]:
        # choose a random reaction
        XS_dict = cell.material.get_all_Macro_XS(energy)
        XS_choices = list(XS_dict.keys())
        XS_values = list(XS_dict.values())

        XS_choice = random.choices(XS_choices, weights=XS_values, k=1)[0]
        return [XS_choice, XS_dict[XS_choice]]
    
    def get_scatter_energy_drop(self, energy:np.float64, A:int) -> np.float64:
        return 0.5 * (1 - ((A-1)/(A+1))**2) * energy

    def track(self):
        # start
        pos = self.source_pos # Start a neutron
        vel = self.get_isotropic_dir()
        energy = self.source_energy

        self.history.append(["i",pos,vel,energy])

        while True:
            # find tracklength of neutron
            current_cell = self.geometry.find_containing_cell(pos) # get the current cell
            tracklength = self.get_tracklength(current_cell, energy) # tracklength
            #print("track length:", tracklength)

            if DELTA_TRACKING == 0:
                # find length to next boundary
                length_to_boundary = self.geometry.distance_to_next_boundary(pos,vel)
                
                if length_to_boundary == None or length_to_boundary <= 1e-14:
                    length_to_boundary = self.geometry.distance_to_next_boundary(pos + 1e-12 * vel,vel)
                #print("boundary length:", length_to_boundary)

                if length_to_boundary == None:
                    self.history.append(["k",pos,vel,energy])
                    break
                    # Neutron attempted to cross outer boundary, break the track

                if tracklength >= length_to_boundary: # Neutron hits boundary
                    pos = pos + length_to_boundary * vel

                    new_cell = self.geometry.find_containing_cell(pos + 1e-6 * vel)
                    # have I entered a larger sphere or smaller one?
                    if new_cell == None:
                        self.history.append(["k",pos,vel,energy])
                        break
                        # Neutron attempted to cross outer boundary, break the track
                    if new_cell.radius < current_cell.radius:
                        if new_cell.boundary_type == 1: #kill boundary
                            self.history.append(["k",pos,vel,energy])
                            break
                            # Neutron is killed, break the track
                        elif new_cell.boundary_type == 2: #reflective boundary
                            vel = self.get_isotropic_dir()
                            self.history.append(["r",pos,vel,energy])
                            # Neutron is scattered off wall, loop
                        else: #transparent
                            self.history.append(["b",pos,vel,energy])
                            # Neutron passes through surface, loop
                    else: 
                        if current_cell.boundary_type == 1: #kill boundary
                            self.history.append(["k",pos,vel,energy])
                            break
                            # Neutron is killed, break the track
                        elif current_cell.boundary_type == 2: #reflective boundary
                            vel = self.get_isotropic_dir()
                            self.history.append(["r",pos,vel,energy])
                            # Neutron is scattered off wall, loop
                        else: #transparent
                            self.history.append(["b",pos,vel,energy])
                            # Neutron passes through surface, loop

                else: # Neutron hits atom
                    pos = pos + tracklength * vel
                    reaction_type = self.get_reaction(current_cell, energy)[0][0]

                    if reaction_type == "a": # absorbtion reaction
                        self.history.append(["a",pos,vel,energy])
                        break
                        # neutron is absorbed, break the track
                    elif reaction_type == "f": # fission reaction
                        self.history.append(["f",pos,vel,energy])
                        break
                        # neutron is absorbed, break the track
                    elif reaction_type == "s": # scatter reaction
                        vel = self.get_isotropic_dir()
                        energy = energy - self.get_scatter_energy_drop(energy, current_cell.material.nuclides[int(self.get_reaction(current_cell, energy)[0][1:])].A)
                        self.history.append(["s",pos,vel,energy])
                    else:
                        raise AttributeError("Unknown Nuclear Reaction!")

            elif DELTA_TRACKING == 1:
                pass

        return self.history

class Run:
    def __init__(self, geometry:Geometry):
        self.geometry = geometry
        self.inactive_k_eff_bank = []
        self.k_eff_bank = []

    def IsotropicSphereSource(self,R:np.float64) -> Vector3D:
        phi = 2 * np.pi * random.random()
        theta = np.arccos(2 * random.random() - 1)
        r = R * random.random()
        return Vector3D(r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta))

    def RussianRouletteArray(self, arr:list, N:int) -> list:
        M = len(arr)
        if N >= M:
            return arr  # No reduction needed
        
        p = 1 - (N / M)  # Probability of removing an element
        while len(arr) > N:
            # Generate a mask using random values and keeping elements with probability 1-p
            arr = [element for element in arr if random.random() >= p]
            
        return arr

    def SplitExtendArray(self, arr: list, N: int) -> list:
        if not isinstance(arr, list):
            raise ValueError("Input must be a list")

        M = len(arr)
        if M >= N:
            return arr  # No upsampling needed

        K = N - M  # Number of elements to add
        additional_samples = random.choices(arr, k=K)  # Sample with replacement
        return arr + additional_samples
    
    def ExtractSourceBank(self, history_bank):
        source_bank = []
        for history in history_bank:
            if history[-1][0] == 'f':
                source_bank.append(history[-1][1])
        return source_bank

    def cycle(self, num_neutrons:int, source_bank:np.ndarray[Vector3D]):
        history_bank = []

        # force the source to match the number of neutrons
        if len(source_bank) > num_neutrons:
            source_bank = self.RussianRouletteArray(source_bank, num_neutrons)
        elif len(source_bank) < num_neutrons:
            source_bank = self.SplitExtendArray(source_bank, num_neutrons)

        # track neutron histories
        for neutron in range(num_neutrons):
            # t1 = time.time()
            neutron_track = Track(self.geometry, source_bank[neutron], E0)
            history = neutron_track.track()
            history_bank.append(history)
            # t2 = time.time()
            # print("Neutron Transport Completed in", round(t2-t1,9),"s")

        return history_bank

    def run(self, num_neutrons:int, inactive_cycles:int, active_cycles:int):
        k_eff_bank = []

        source_bank = []
        for i in range(num_neutrons):
            source_bank.append(self.IsotropicSphereSource(self.geometry.root.radius))

        for i in range(inactive_cycles):
            source_bank = self.ExtractSourceBank(self.cycle(num_neutrons, source_bank))
            self.inactive_k_eff_bank.append(nubar * len(source_bank) / num_neutrons)
            print("Inactive Cycle:\t", i, "\tk_eff =", round(self.inactive_k_eff_bank[-1],4))

        history_bank_bank = []
        for j in range(active_cycles):
            history_bank = self.cycle(num_neutrons, source_bank)
            history_bank_bank.append(history_bank)
            source_bank = self.ExtractSourceBank(history_bank)
            self.k_eff_bank.append(nubar * len(source_bank) / num_neutrons)
            print("Active Cycle:\t", i+j+1, "\tk_eff =", round(self.k_eff_bank[-1],4), "\t±", round(self.k_eff_bank[-1]/np.sqrt(num_neutrons),4))

    # def plot_source_distribution(self,)

    def plot_k_eff(self):
        # Extract inactive and active k_eff banks
        inactive_k_eff = self.inactive_k_eff_bank
        active_k_eff = self.k_eff_bank

        # Create index arrays
        inactive_indices = np.arange(len(inactive_k_eff))
        active_indices = np.arange(len(inactive_k_eff), len(inactive_k_eff) + len(active_k_eff))

        # Create figure
        plt.figure(figsize=(8, 5))

        # 1. Plot the connecting line FIRST so it's underneath
        if len(inactive_k_eff) > 0 and len(active_k_eff) > 0:
            plt.plot(
                [inactive_indices[-1], active_indices[0]],  # x-coordinates
                [inactive_k_eff[-1], active_k_eff[0]],      # y-coordinates
                linestyle='-', color='b', zorder=1  # Lower z-order so it's underneath
            )

        # 2. Plot active points in blue
        plt.plot(active_indices, active_k_eff, marker='o', linestyle='-', color='b', label="Active Cycles", zorder=2)

        # 3. Plot inactive points in red (on top)
        plt.plot(inactive_indices, inactive_k_eff, marker='o', linestyle='-', color='r', label="Inactive Cycles", zorder=3)

        # Labels and title
        plt.xlabel("Cycle Number")
        plt.ylabel(r"$k_{\text{eff}}$")
        plt.title(r"$k_{\text{eff}}$ vs Cycle")
        plt.legend()
        plt.grid(True)

        # plt.show()
        plt.savefig("k_eff.png")
        plt.close()

    def history_quiver(self, history_bank:list):
        """3D quiver plot of multiple neutron histories"""

        def generate_sphere(center, radius, num_points=20):
            u = np.linspace(0, 2 * np.pi, num_points)
            v = np.linspace(0, np.pi, num_points)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
            return x, y, z

        def normalize_energy(energy, min_energy, max_energy):
            return (energy - min_energy) / (max_energy - min_energy) if max_energy > min_energy else 0.5

        arrow_head_size = 0.1  # Adjust this value to change arrowhead size
        arrow_width = 1        # Adjust this for thicker arrows
        point_scale = 1

        collision_colormap = {
            "s": "blue",      # Scatter
            "a": "red",       # Absorption
            "f": "green",     # Fission
            "k": "black",     # Leakage
            "r": "black",     # Reflection
            "i": "black",      # Initial point
            "b": "grey"
        }

        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 3D geometry Plotter

        for cell in self.geometry.cells:
            x, y, z = generate_sphere(cell.center, cell.radius)
            ax.plot_wireframe(x, y, z, color="gray", alpha=0.5, linewidth=0.5)  # Light gray background spheres

        # History Plotter
        for history in history_bank:

            energies = [entry[3] for entry in history]
            min_energy, max_energy = 0, max(energies)

            cmap = plt.cm.plasma
            norm = mcolors.Normalize(vmin=min_energy, vmax=max_energy)

            collision_labels = {
                "s": "Scatter",
                "a": "Absorption",
                "f": "Fission",
                "k": "Kill Boundary",
                "r": "Reflection Boundary",
                "i": "Source",
                "b": "Boundary crossing"
            }
            collision_handles = {}

            for i in range(len(history) - 1):
                ctype, pos, direction, energy = history[i]
                next_pos = history[i + 1][1]
                next_ctype = history[i + 1][0]

                # Normalize energy for color mapping
                color = cmap(normalize_energy(energy, min_energy, max_energy))

                # Plot collision points
                scatter = ax.scatter(*pos, color=collision_colormap.get(ctype, "gray"), s=point_scale)

                if i == len(history) - 2:
                    scatter = ax.scatter(*next_pos, color=collision_colormap.get(next_ctype, "gray"), s=point_scale)

                # Store for legend (ensures unique labels)
                if ctype not in collision_handles:
                    collision_handles[ctype] = scatter

                # Compute direction vector for quiver
                direction_vector = next_pos - pos  # Keep arrow length unchanged

                # Plot arrows with custom arrowhead size
                ax.quiver(*pos, *direction_vector, color=color, linewidth=arrow_width, arrow_length_ratio=arrow_head_size)


        # Set equal scaling for all axes
        max_range = np.max([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]])
        min_range = np.min([ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]])

        # Set equal limits for x, y, z axes
        ax.set_xlim([min_range, max_range])
        ax.set_ylim([min_range, max_range])
        ax.set_zlim([min_range, max_range])

        # Create collision type legend
        # Create collision type legend with custom labels
        # Create collision type legend with custom labels
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color, label=collision_labels[ctype])
            for ctype, color in collision_colormap.items()
        ]

        # Create the legend with human-readable labels
        ax.legend(handles=legend_handles, title="Collision Types", loc="upper left", bbox_to_anchor=(-0.3, 1))


        # Add a colorbar for energy scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label("Neutron Energy (eV)")

        ax.set_xlabel("X Position (cm)")
        ax.set_ylabel("Y Position (cm)")
        ax.set_zlabel("Z Position (cm)")
        ax.set_title("3D Neutron Transport History")

        plt.savefig("Neutron_Transport_History.png")
        plt.show()
        plt.close()

header = '''===========================================
        _____      _     __  __  _____ 
       / ____|    | |   |  \/  |/ ____|
      | (___   ___| |__ | \  / | |     
       \___ \ / _ \ '_ \| |\/| | |     
       ____) |  __/ |_) | |  | | |____ 
      |_____/ \___|_.__/|_|  |_|\_____|

 A budget Monte Carlo neutron transport code
 
 ==========================================='''

print(header)

class GeometryGenerator:
    @staticmethod
    def generate_packed_spheres_SampleReject(large_center, large_radius, small_radius, packing_factor) -> Geometry:
        """
        Generates a geometry consisting of a large sphere packed with many smaller spheres,
        reaching a desired packing factor.

        Parameters:
        - large_center: Vector3D -> Center of the large sphere.
        - large_radius: float -> Radius of the large sphere.
        - small_radius: float -> Radius of each small sphere.
        - packing_factor: float -> Desired volume fraction of small spheres inside the large sphere.

        Returns:
        - Geometry object with a tree structure containing the packed spheres.
        """
        large_volume = (4/3) * np.pi * (large_radius ** 3)
        small_volume = (4/3) * np.pi * (small_radius ** 3)

        # Required number of small spheres to achieve the packing factor
        target_num_spheres = int((packing_factor * large_volume) / small_volume)

        # Create the large sphere
        u235 = Nuclide("U235",0.1e23,np.asarray([1,2,3]))
        mat = Material(np.asarray([u235]),300)
        large_sphere = Cell(0, 0, 1, mat, large_center, large_radius)

        small_spheres: List[Cell] = []
        attempts = 0  # Track how many placement attempts are made

        while len(small_spheres) < target_num_spheres and attempts < target_num_spheres * 10:
            attempts += 1

            # Generate a random position inside the large sphere
            rand_dir = np.random.normal(size=3)
            rand_dir /= np.linalg.norm(rand_dir)  # Normalize to unit vector
            rand_dist = (large_radius - small_radius) * np.cbrt(random.uniform(0, 1))
            rand_position = large_center + rand_dir * rand_dist

            # Ensure no overlap with existing small spheres
            overlap = any(np.linalg.norm(rand_position - np.array(sphere.center)) < 2 * small_radius
                          for sphere in small_spheres)

            if not overlap:
                new_sphere = Cell(len(small_spheres) + 1, 0, 1, mat, rand_position, small_radius)
                small_spheres.append(new_sphere)
                large_sphere.children.append(new_sphere)  # Add as child to tree

        print(f"Generated {len(small_spheres)} spheres (Target: {target_num_spheres})")

        return Geometry([large_sphere] + small_spheres)
    
    @staticmethod
    def generate_packed_spheres_RandomWalk(large_center, large_radius, small_radius, packing_factor) -> Geometry:
        """
        Generates a geometry consisting of a large sphere packed with many smaller spheres,
        reaching a desired packing factor using random walk + local minimization.
        
        Parameters:
        - large_center: Vector3D -> Center of the large sphere.
        - large_radius: float -> Radius of the large sphere.
        - small_radius: float -> Radius of each small sphere.
        - packing_factor: float -> Desired volume fraction of small spheres inside the large sphere.
        
        Returns:
        - Geometry object with a tree structure containing the packed spheres.
        """
        large_volume = (4/3) * np.pi * (large_radius ** 3)
        small_volume = (4/3) * np.pi * (small_radius ** 3)
        
        # Required number of small spheres to achieve the packing factor
        target_num_spheres = int((packing_factor * large_volume) / small_volume)

        # Create the large sphere
        u235 = Nuclide("U235",0.1e23,np.asarray([1,2,3]))
        mat = Material(np.asarray([u235]),300)
        large_sphere = Cell(0, 0, 1, mat, large_center, large_radius)

        small_spheres: List[Cell] = []
        
        attempts = 0
        max_attempts = target_num_spheres * 20
        max_iterations = 500  # Maximum number of relaxation iterations
        sphere_positions = []

        # Generate spheres with random positions
        while len(small_spheres) < target_num_spheres and attempts < max_attempts:
            attempts += 1

            # Generate a random position inside the large sphere
            rand_dir = np.random.normal(size=3)
            rand_dir /= np.linalg.norm(rand_dir)  # Normalize to unit vector
            rand_dist = (large_radius - small_radius) * np.cbrt(random.uniform(0, 1))
            rand_position = large_center + rand_dir * rand_dist

            # Relaxation phase to minimize overlap
            for _ in range(max_iterations):
                # Check for overlap with existing spheres
                overlap = any(np.linalg.norm(rand_position - np.array(sphere.center)) < 2 * small_radius
                              for sphere in small_spheres)
                
                if not overlap:
                    # Add sphere to the geometry if no overlap
                    new_sphere = Cell(len(small_spheres) + 1, 0, 1, mat, rand_position, small_radius)
                    small_spheres.append(new_sphere)
                    large_sphere.children.append(new_sphere)
                    sphere_positions.append(rand_position)
                    break  # Break after successfully adding a sphere
                else:
                    # Push the sphere away to avoid overlap (relaxation)
                    rand_position += np.random.normal(scale=0.01, size=3)

        print(f"Generated {len(small_spheres)} spheres (Target: {target_num_spheres})")
        return Geometry([large_sphere] + small_spheres)
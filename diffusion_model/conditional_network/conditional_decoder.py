import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append("/workspace/diffusion_model")
sys.path.append("/workspace")
import torch
from typing import List
from sphere_metasurface.parametrization import Sphere_surface
from dataset_creation import extrapolate_params, calculate_field
import numpy as np
import matplotlib.pyplot as plt


class Decoder_conditional_diffusional_vector():
    def __init__(self, 
                 vector_to_decode: torch.Tensor,
                 angles: List[float],
                 number_of_cells: int,
                 side_length: float,
                 reflective_index: complex,
                 vacuum_wavelength: float,
                 polar_angle: float = np.pi,  # angle in radians, pi == from top
                 azimuthal_angle: float = 0,  # angle in radians, 0 == x-axis
                 polarization: int = 0,  # 0 for TE, 1 for TM polarization
                 conditional_dscs_surface: List[float] = None
                 ):
        self.vector_to_decode = vector_to_decode.tolist()[0][0]
        self.angles = angles
        self.number_of_cells = number_of_cells
        self.side_length = side_length
        self.reflective_index = reflective_index
        self.vacuum_wavelength = vacuum_wavelength
        self.polar_angle = polar_angle
        self.azimuthal_angle = azimuthal_angle
        self.polarization = polarization
        self.conditional_dscs_surface = conditional_dscs_surface
        
    def compute_sphere_surface(self):
        sphere_surface = Sphere_surface(number_of_cells=self.number_of_cells, 
                                    side_length=self.side_length,
                                    reflective_index=self.reflective_index)
        sphere_surface.mesh_generation()
        
        sphere_coordinates_01 = self.vector_to_decode[:2*self.number_of_cells**2]
        sphere_radiuses_01 = self.vector_to_decode[2*self.number_of_cells**2:2*self.number_of_cells**2 + self.number_of_cells**2]

        parameters_01 = []
        for i in range(len(sphere_radiuses_01)):
            parameters_01.append(sphere_coordinates_01[2*i])
            parameters_01.append(sphere_coordinates_01[2*i+1])
            parameters_01.append(sphere_radiuses_01[i])

        real_params = extrapolate_params(parameters_01, sphere_surface)
        real_x = real_params[::3]
        real_y = real_params[1::3]
        real_radiuses = real_params[2::3]
        real_coordinates = [[real_x[i], real_y[i], 0] for i in range(len(real_x))]

        sphere_surface.__spheres_add__(real_radiuses, real_coordinates)
        dscs_surface = calculate_field(spheres_surface=sphere_surface, 
                                       vacuum_wavelength=self.vacuum_wavelength, 
                                       polar_angle=self.polar_angle, 
                                       azimuthal_angle=self.azimuthal_angle, 
                                       polarization=self.polarization)
        
        self.angles_indices = []
        self.all_angles = np.arange(0, 180, 0.5)
        for angle in self.angles:
            idx = np.argmin(np.abs(self.all_angles - angle))
            self.angles_indices.append(idx)
        self.dscs_surface = [dscs_surface[i] for i in self.angles_indices]
    
    def compute_loss(self):
        loss_mpe = (100/len(self.dscs_surface))*np.abs(np.sum([(self.dscs_surface[i] - self.conditional_dscs_surface[i])/self.dscs_surface[i] for i in range(len(self.dscs_surface))]))
        return loss_mpe
    
    # def plot_dscs(self):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.all_angles, self.dscs_surface, label='Generated DSCS')
    #     plt.plot(self.angles[self.angles_indices], self.conditional_dscs_surface, label='Conditional DSCS')
    #     plt.legend()
    #     plt.show()

if __name__ == "__main__":
    vector_to_decode = torch.tensor([[[0.3029, 0.7002, 0.3299, 0.3194, 0.0410, 0.9797, 0.4327, 0.7123,
          0.1524, 0.1348, 0.3902, 0.0826, 0.0242, 0.0165, 0.0101, 0.0373]]])
    decoder = Decoder_conditional_diffusional_vector(vector_to_decode, 
                                         angles = [0, 10, 20, 40, 60, 80, 100, 120, 140, 160],
                                         number_of_cells = 2,
                                         side_length = 10,
                                         reflective_index = 2,
                                         vacuum_wavelength = 1,
                                         conditional_dscs_surface=[0.3, 0.1, 0.05, 0.03, 0.01, 0.04, 0.08, 0.1, 0.09, 0.15])
    dscs_surface = decoder.compute_sphere_surface()

    loss = decoder.compute_loss()
    
    print(loss)
import smuthi.simulation
import smuthi.particles
import smuthi.initial_field
import smuthi.layers
import smuthi.postprocessing.far_field as ff

import numpy as np

from parametrization import Sphere_surface


def calculate_loss(spheres_surface, 
                   object, 
                   vacuum_wavelength: float,
                   angles_to_mimic: list,
                   polar_angle: float = np.pi,  # angle in radians, pi == from top
                   azimuthal_angle: float = 0,  # angle in radians, 0 == x-axis
                   polarization: int = 0  # 0 for TE, 1 for TM polarization
                   ):
    """
    Calculate the loss value for the given spheres surface, object, initial field, and angle to mimicking
    """
    layers = smuthi.layers.LayerSystem()

    initial_field = smuthi.initial_field.PlaneWave(vacuum_wavelength=vacuum_wavelength,
                                                   polar_angle=polar_angle,
                                                   azimuthal_angle=azimuthal_angle,
                                                   polarization=polarization)
    
    # Create particle lists
    surface_particles = spheres_surface.spheres
    object_particles = object 
    
    # Set parameters for all particles
    for particle in surface_particles:
        particle.l_max = 3  # multipolar order
        particle.m_max = 3  # azimuthal order

    for particle in object_particles:
        particle.l_max = 3  # multipolar order
        particle.m_max = 3  # azimuthal order
    
    # Create and run simulation for surface
    simulation_surface = smuthi.simulation.Simulation(layer_system=layers,
                                                    particle_list=surface_particles,
                                                    initial_field=initial_field)
    simulation_surface.run()

    # Create and run simulation for object
    simulation_object = smuthi.simulation.Simulation(layer_system=layers,
                                                   particle_list=object_particles,
                                                   initial_field=initial_field)
    simulation_object.run()


    far_field_surface = ff.scattered_far_field(vacuum_wavelength=vacuum_wavelength,
                                  particle_list=surface_particles,
                                  layer_system=layers, 
                                  polar_angles=np.array(angles_to_mimic),
                                  )
    
    far_field_object = ff.scattered_far_field(vacuum_wavelength=vacuum_wavelength,
                                  particle_list=object_particles,
                                  layer_system=layers, 
                                  polar_angles=np.array(angles_to_mimic),
                                  )
    
    dscs_surface = np.sum(far_field_surface.azimuthal_integral(), axis=0) * np.pi / 180
    dscs_object = np.sum(far_field_object.azimuthal_integral(), axis=0) * np.pi / 180
    
    # loss_value = np.sum(np.abs(dscs_surface - dscs_object))/np.mean(dscs_object)

    loss_value = np.mean((dscs_surface - dscs_object)**2)
    return loss_value

if __name__ == "__main__":

    spheres_surface = Sphere_surface(
        number_of_cells=3,
        side_length=3.0,
        reflective_index=complex(2.0, 0.0)
    )

    coordinates_list = [
        [0.3, 0.5, 0],  
        [1.5, 0.4, 0],  
        [2.2, 0.5, 0],  
        [0.7, 1.5, 0],  
        [1.5, 1.4, 0],
        [2.55, 1.5, 0],
        [0.45, 2.45, 0],
        [1.6, 2.7, 0],
        [2.5, 2.4, 0]
    ]
    
    spheres_radius_list = [0.1] * 9
    
    spheres_surface.__spheres_add__(spheres_radius_list, coordinates_list)

    object = [smuthi.particles.Sphere(radius=0.1, position=[0, 0, 0], refractive_index=complex(5, 0.0))]

    loss = calculate_loss(spheres_surface, object, 0.5, [np.pi/4, np.pi/3])
    
    print(loss)


# Optimization(spheres_surface, object) -> sphere_radius_list, coordinates_list 
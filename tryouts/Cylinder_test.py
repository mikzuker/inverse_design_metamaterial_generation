#*****************************************************************************#
# This is a simple example script for Smuthi v1.0.0                           #
# It evaluates the electric near field for three spheres in a waveguide       #
# excited by a plane wave under oblique incidence                             #
#*****************************************************************************#

import numpy as np
import smuthi.simulation
import smuthi.initial_field
import smuthi.layers
import smuthi.particles
import smuthi.postprocessing.graphical_output as go
import smuthi.utility.cuda
import smuthi.postprocessing.far_field as ff


smuthi.utility.cuda.enable_gpu()

layers = smuthi.layers.LayerSystem()

particle1 = smuthi.particles.FiniteCylinder(position=[0, 0, 250],  # Подняли вверх
                                           refractive_index=2, 
                                           cylinder_radius=15,
                                           cylinder_height=50,
                                           euler_angles=[0, 0, 0],
                                           l_max=3)

particle2 = smuthi.particles.FiniteCylinder(position=[50, 0, 250],  # Подняли вверх
                                           refractive_index=2, 
                                           cylinder_radius=15,
                                           cylinder_height=50,
                                           euler_angles=[0, 0, 0],
                                           l_max=3)

# particle = smuthi.particles.Sphere(position=[0,0,0],
#                                  refractive_index=4+1j,
#                                  radius=200,
#                                  l_max=4)


particles = [particle1, particle2]

# Initial field
plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=20,
                                            polar_angle=np.pi, # from top
                                            azimuthal_angle=0,
                                            polarization=0)         # 0=TE 1=TM

# Initialize and run simulation
simulation = smuthi.simulation.Simulation(layer_system=layers,
                                          particle_list=particles,
                                          initial_field=plane_wave,
                                          length_unit='nm')
simulation.run()

# Create plots that visualize the electric near field.

# go.show_near_field(quantities_to_plot=['norm(E)'],
#                    show_plots=True,
#                    show_opts=[{'interpolation':'quadric'}],
#                    save_plots=True,
#                    save_opts=[{'format':'png'}], 
#                    outputdir='./output',
#                    xmin=-200,
#                    xmax=200,
#                    zmin=-100,
#                    zmax=300,
#                    resolution_step=20,
#                    simulation=simulation,
#                    show_internal_field=True)

# go.show_near_field(quantities_to_plot=['norm(E)'],
#                    show_plots=True,
#                    show_opts=[{'interpolation':'quadric'}],
#                    save_plots=True,
#                    save_data=True,
#                    save_opts=[{'format':'png'}], 
#                    outputdir='./output',
#                    xmin=-200,
#                    xmax=200,
#                    zmin=-100,
#                    zmax=300,
#                    resolution_step=20,
#                    simulation=simulation,
#                    show_internal_field=True)

# go.show_scattered_far_field(simulation, show_plots=True, show_opts=[{'label':'scattered_far_field'}],
                            #  save_plots=True, save_opts=None,
                            #  save_data=False, data_format='hdf5', outputdir='./output',
                            #  flip_downward=True, split=True, log_scale=False,
                            #  polar_angles='default', azimuthal_angles='default', angular_resolution=None)

# scs = ff.total_scattering_cross_section(initial_field=plane_wave,
#                                         particle_list=particles,
#                                         layer_system=layers)

far_field_from_object = ff.scattered_far_field(vacuum_wavelength=20,
                                       particle_list=particles,
                                       layer_system=layers,
                                       polar_angles='default',
                                       azimuthal_angles='default')
                                       

print("\n****************************************************")
# print("Scattering cross section: %e µm^2"%(scs/1e6))
print(far_field_from_object)
print("****************************************************")



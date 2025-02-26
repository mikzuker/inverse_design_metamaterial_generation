import numpy as np
import smuthi.particles
import smuthi.initial_field
import smuthi.layers
import smuthi.simulation
import smuthi.postprocessing.far_field as ff
import smuthi.postprocessing.graphical_output as go
import matplotlib.pyplot as plt

layers = smuthi.layers.LayerSystem()

sphere = smuthi.particles.Sphere(position=[0,0,0],
                                 refractive_index=4+1j,
                                 radius=100,
                                 l_max=4)

one_sphere = [sphere]

scattering = []

for wavelength in range(200, 701, 10): 
    plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=wavelength,
                                            polar_angle=np.pi,    
                                            azimuthal_angle=0,   
                                            polarization=0)         

    simulation = smuthi.simulation.Simulation(layer_system=layers,
                                          particle_list=one_sphere,
                                          initial_field=plane_wave)
    simulation.run()

    scs = ff.total_scattering_cross_section(initial_field=plane_wave,
                                        particle_list=one_sphere,
                                        layer_system=layers)
    
    scattering.append(scs/(np.pi*100**2))

print(scattering)

plt.plot(range(200, 701, 10), scattering)
# plt.savefig('1D_scattering')
go.show_near_field(quantities_to_plot=['norm(E)'],
                   show_plots=True,
                   show_opts=[{'interpolation': 'quadric'}],
                   save_plots=True,
                   save_opts=[{'format': 'pdf', 'dpi': 250}],
                   outputdir='./output',
                   xmin=-200,
                   xmax=200,
                   zmin=-200,
                   zmax=200,
                   resolution_step=30,
                   simulation=simulation,
                   show_internal_field=True)
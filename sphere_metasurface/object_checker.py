import smuthi 
import smuthi.postprocessing.far_field as ff
import smuthi.initial_field 
import smuthi.simulation 
import smuthi.layers as layers
import smuthi.particles as particles

import matplotlib.pyplot as plt
import numpy as np


layers = smuthi.layers.LayerSystem()

initial_field = smuthi.initial_field.PlaneWave(vacuum_wavelength=0.5,
                                                   polar_angle=np.pi,
                                                   azimuthal_angle=0,
                                                   polarization=0)
    
object_particles = [smuthi.particles.FiniteCylinder(position=[0, 0, 250],  # Подняли вверх
                                           refractive_index=2, 
                                           cylinder_radius=15,
                                           cylinder_height=50,
                                           euler_angles=[0, 0, 0],
                                           l_max=3)]

# for particle in object_particles:
#         particle.l_max = 3  # multipolar order
#         particle.m_max = 3  # azimuthal order

simulation_object = smuthi.simulation.Simulation(layer_system=layers,
                                                particle_list=object_particles,
                                                initial_field=initial_field)
simulation_object.run()
    
whole_far_field_object = ff.scattered_far_field(vacuum_wavelength=0.5,
                                particle_list=object_particles,
                                layer_system=layers 
                                )
    
whole_dscs_object = np.sum(whole_far_field_object.azimuthal_integral(), axis=0) * np.pi / 180

fig = plt.figure(figsize=(10, 10))
object_array = [np.arange(0, 360, 1), whole_dscs_object]
plt.plot(*object_array, label='object', linewidth=2)
                        
plt.legend()
plt.xlabel('Angle')
plt.ylabel('DSCS')
plt.title('Spectrum')
plt.grid()
plt.yscale('log')
plt.savefig('/workspace/sphere_metasurface/plots/object_try.pdf')
plt.close()

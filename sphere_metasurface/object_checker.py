import matplotlib.pyplot as plt
import numpy as np
import smuthi
import smuthi.initial_field
import smuthi.layers
import smuthi.postprocessing.far_field as ff
import smuthi.simulation

layers = smuthi.layers.LayerSystem()

initial_field = smuthi.initial_field.PlaneWave(
    vacuum_wavelength=2, polar_angle=np.pi, azimuthal_angle=0, polarization=0
)

object_particles = [
    smuthi.particles.FiniteCylinder(
        position=[0, 0, 0],
        refractive_index=2,
        cylinder_radius=5,
        cylinder_height=2,
        euler_angles=[0, 0, 0],
        l_max=5,
    )
]
# smuthi.particles.FiniteCylinder(position=[0, 200, 0],
#                        refractive_index=4,
#                        cylinder_radius=50,
#                        cylinder_height=20,
#                        euler_angles=[0, 0, 0],
#                        l_max=3)]

# for particle in object_particles:
#         particle.l_max = 3  # multipolar order
#         particle.m_max = 3  # azimuthal order

simulation_object = smuthi.simulation.Simulation(
    layer_system=layers, particle_list=object_particles, initial_field=initial_field
)
simulation_object.run()

whole_far_field_object = ff.scattered_far_field(
    vacuum_wavelength=2, particle_list=object_particles, layer_system=layers
)

whole_dscs_object = (
    np.sum(whole_far_field_object.azimuthal_integral(), axis=0) * np.pi / 180
)

fig = plt.figure(figsize=(10, 10))
object_array = [np.arange(0, 180, 0.5), whole_dscs_object]
plt.plot(*object_array, label="object", linewidth=2)

plt.legend()
plt.xlabel("Angle")
plt.ylabel("DSCS")
plt.title("Spectrum")
plt.grid()
# plt.yscale('log')
plt.savefig("/workspace/sphere_metasurface/plots/object_try_13.pdf")
plt.close()

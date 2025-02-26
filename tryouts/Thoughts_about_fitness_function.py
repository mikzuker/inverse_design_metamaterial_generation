from sphere_metasurface.parametrization import Sphere_surface
import numpy as np
import smuthi.postprocessing.scattered_field as sf
import smuthi.postprocessing.internal_field as intf
import smuthi.postprocessing.far_field as ff
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from itertools import cycle
import tempfile
import shutil
#import imageio
import os
import warnings
import sys
from tqdm import tqdm
from smuthi.postprocessing.graphical_output import show_far_field
import smuthi.particles
import smuthi.initial_field
import smuthi.layers
import smuthi.simulation
import numpy as np
import matplotlib.pyplot as plt
import smuthi.postprocessing.graphical_output as go

def show_scattered_far_field(simulation, show_plots=True, show_opts=[{'label':'scattered_far_field'}],
                             save_plots=False, save_opts=None,
                             save_data=False, data_format='hdf5', outputdir='.',
                             flip_downward=True, split=True, log_scale=False,
                             polar_angles='default', azimuthal_angles='default', angular_resolution=None):
    """Display and export the scattered far field.

    Args:
        simulation (smuthi.simulation.Simulation):  Simulation object
        show_plots (bool):                          Display plots if True
        show_opts (dict list):  List of dictionaries containing options to be passed to pcolormesh for plotting.
                                If save_plots=True, a 1:1 correspondence between show_opts and save_opts dictionaries
                                is assumed. For simplicity, one can also provide a single show_opts entry that will
                                be applied to all save_opts.
                                The following keys are available (see matplotlib.pyplot.pcolormesh documentation):
                                'alpha'     (None)
                                'cmap'      ('inferno')
                                'norm'      (None), is set to matplotlib.colors.LogNorm() if log_scale is True
                                'vmin'      (None), applies only to 2D plots
                                'vmax'      (None), applies only to 2D plots
                                'shading'   ('nearest'), applies only to 2D plots. 'gouraud' is also available
                                'linewidth' (None), applies only to 1D plots
                                'linestyle' (None), applies only to 1D plots
                                'marker'    (None), applies only to 1D plots
                                An optional extra key called 'label' of type string is shown in the plot title
                                and appended to the associated file if save_plots is True
        save_plots (bool):      If True, plots are exported to file.
        save_opts (dict list):  List of dictionaries containing options to be passed to savefig.
                                A 1:1 correspondence between save_opts and show_opts dictionaries is assumed. For
                                simplicity, one can also provide a single save_opts entry that will be applied to
                                all show_opts.
                                The following keys are made available (see matplotlib.pyplot.savefig documentation):
                                'dpi'           (None)
                                'orientation'   (None)
                                'format'        ('png'), also available: eps, jpeg, jpg, pdf, ps, svg, tif, tiff ...
                                'transparent'   (False)
                                'bbox_inches'   ('tight')
                                'pad_inches'    (0.1)
        save_data (bool):       If True, raw data are exported to file
        data_format (str):      Output data format string, 'hdf5' and 'ascii' formats are available
        outputdir (str):                        Path to the directory where files are to be saved
        flip_downward (bool):                   If True, represent downward directions as 0-90 deg instead of 90-180
        split (bool):                           If True, show two different plots for upward and downward directions
        log_scale (bool):                       If True, set a logarithmic scale
        polar_angles (numpy.ndarray or str):    Polar angles values (radian).
                                                If 'default', use smuthi.fields.default_polar_angles
        azimuthal_angles (numpy.ndarray or str):Azimuthal angle values (radian).
                                                If 'default', use smuthi.fields.default_azimuthal_angles
        angular_resolution (float):             If provided, angular arrays are generated with this angular resolution
                                                over the default angular range
    """

    infld = simulation.initial_field
    plst = simulation.particle_list
    lsys = simulation.layer_system
    far_field = ff.scattered_far_field(vacuum_wavelength=infld.vacuum_wavelength,
                                       particle_list=plst,
                                       layer_system=lsys,
                                       polar_angles=polar_angles,
                                       azimuthal_angles=azimuthal_angles,
                                       angular_resolution=angular_resolution)

    [d.setdefault('label','scattered_far_field') for d in show_opts]

    show_far_field(far_field=far_field, save_plots=save_plots, save_opts=save_opts, show_plots=show_plots,
                   show_opts=show_opts, save_data=save_data, data_format=data_format, outputdir=outputdir,
                   flip_downward=flip_downward, split=split, log_scale=log_scale)

def get_far_field_data(far_field, flip_downward=True):
    """Extract far field intensity data for plotting.
    
    Args:
        far_field: SMUTHI far field object
        flip_downward: If True, represent downward directions as 0-90 deg instead of 90-180
    
    Returns:
        tuple: (angles, intensities) where angles in degrees and intensities are normalized
    """
    # Get polar angles in degrees
    angles = np.degrees(far_field.polar_angles)
    
    # Calculate intensities (|E_theta|^2 + |E_phi|^2)
    intensities = np.sum(np.abs(far_field.E_field)**2, axis=0).squeeze()
    
    # Flip angles for downward direction if requested
    if flip_downward:
        downward = angles > 90
        angles[downward] = 180 - angles[downward]
    
    return angles, intensities


surface = Sphere_surface(
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
    
surface.__spheres_add__(spheres_radius_list, coordinates_list)
    
    
layers = smuthi.layers.LayerSystem()
    
plane_wave = smuthi.initial_field.PlaneWave(vacuum_wavelength=0.5,
                                                polar_angle=np.pi,
                                                azimuthal_angle=0,
                                                polarization=0)
    
simulation = smuthi.simulation.Simulation(layer_system=layers,
                                              particle_list=surface.spheres,
                                              initial_field=plane_wave,
                                              length_unit='nm')
    
simulation.run()

# Calculate far field
far_field = ff.scattered_far_field(vacuum_wavelength=0.5,
                                  particle_list=surface.spheres,
                                  layer_system=layers,
                                #   polar_angles=np.linspace(0, np.pi/2, 91) 
                                  polar_angles=np.array([0, np.pi/4]))

# print(np.abs(far_field.amplitude)**2)
# Get data for plotting
# angles, intensities = get_far_field_data(far_field, flip_downward=True)
print(far_field.polar_angles * 180 / np.pi)
print(np.sum(far_field.azimuthal_integral(), axis=0) * np.pi / 180)
# # Plot the far field pattern
# plt.figure()
# plt.plot(far_field.polar_angles * 180 / np.pi, np.sum(far_field.azimuthal_integral(), axis=0) * np.pi / 180)
# plt.xlabel('Angle (degrees)')
# plt.ylabel('Intensity')
# plt.title('Far Field Pattern')
# plt.savefig('/workspace/sphere_metasurface/plots/far_field_pattern.png')
# plt.close()
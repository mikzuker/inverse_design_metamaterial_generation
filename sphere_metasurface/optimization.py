import json
from pathlib import Path
import numpy as np
import random
import cmaes
from tqdm import tqdm
import matplotlib.pyplot as plt

import smuthi.particles as particles

from parametrization import Sphere_surface
from fitness_function import calculate_loss, calculate_spectrum

class Optimization(object):
    def __init__(self,
                 object_to_mimic, 
                 vacuum_wavelength: float,
                 angeles_to_mimic: list,
                 side_length: float,
                 number_of_cells: int,
                 refractive_index: complex,
                 iterations: int,
                 seed: int,
                 ):
        self.surface = Sphere_surface(number_of_cells, side_length, refractive_index)
        self.object_to_mimic = object_to_mimic
        self.vacuum_wavelength = vacuum_wavelength
        self.angeles_to_mimic = angeles_to_mimic
        self.side_length = side_length
        self.number_of_cells = number_of_cells
        self.refractive_index = refractive_index
        self.iterations = iterations
        self.seed = seed

    def extrapolate_params(self, params):
            real_params = []
            n_spheres = len(params) // 3
            
            for i in range(len(self.surface.squares)):
                square = self.surface.squares[i]
                cell_size = square[1][0] - square[0][0]  # размер ячейки
                
                # Вычисляем реальный радиус (0 -> 0, 1 -> cell_size/2)
                radius = params[i*3 + 2] * (cell_size/2)
                
                # Вычисляем реальные координаты с учетом радиуса
                x_min = square[0][0] + radius  # левая граница + радиус
                x_max = square[1][0] - radius  # правая граница - радиус
                x = x_min + params[i*3] * (x_max - x_min)  # интерполяция по x
                
                y_min = square[0][1] + radius  # нижняя граница + радиус
                y_max = square[1][1] - radius  # верхняя граница - радиус
                y = y_min + params[i*3 + 1] * (y_max - y_min)  # интерполяция по y
                
                real_params.extend([x, y, radius])
            
            return real_params
    
    def optimize(self):
        """
        Optimize the sphere surface using CMA-ES algorithm.
        """
        self.surface.mesh_generation()

        n_spheres = self.number_of_cells**2
        random.seed(self.seed)
        initial_params = [random.uniform(0, 1) for _ in range(3*n_spheres)]
        
        cell_size = self.surface.side_length / self.surface.number_of_cells

        def objective_function(params):
          
            real_params = self.extrapolate_params(params)
            real_x = real_params[::3]
            real_y = real_params[1::3]
            real_coordinates_list = [[real_x[i], real_y[i], 0] for i in range(len(real_x))]
            
         
            surface = Sphere_surface(self.number_of_cells, self.side_length, self.refractive_index)
            surface.__spheres_add__(coordinates_list=real_coordinates_list, spheres_radius_list=real_params[2::3])
            
            loss_value = calculate_loss(surface, 
                                        self.object_to_mimic, 
                                        self.vacuum_wavelength, 
                                        self.angeles_to_mimic
                                        )
            
            return loss_value

        population_size = 16
        # Запускаем CMA-ES
        opts = cmaes.CMA(mean=np.array(initial_params), 
                         sigma=0.1 * cell_size,
                         bounds=np.tile([0, 1], (len(initial_params), 1)),
                         seed=self.seed,
                         population_size=population_size)
        cnt = 0
        
        max_value, max_params = 100000, []

        pbar = tqdm(range(self.iterations))
        progress = []

        for generation in pbar:
            solutions = []
            values = []
            for _ in range(population_size):
                params = opts.ask()
                value = objective_function(params)

                values.append(value)
                if value < max_value:
                    max_value = value
                    max_params = params
                    cnt += 1

                solutions.append((params, value))
           
            opts.tell(solutions)
            progress.append(np.around(np.mean(values), 15))

        pbar.set_description(
            "Processed %s generation\t max %s mean %s"
            % (generation, np.around(max_value, 15), np.around(np.mean(values), 15))
        )


        results = {
            "params": max_params,
            "optimized_value": max_value,
            "progress": progress,
        }

        experiment_dir = Path("sphere_metasurface/results") / f"experiment_{self.side_length}_{self.number_of_cells}_{self.number_of_cells}_{self.refractive_index}_{self.seed}_{self.iterations}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # def save_results(results):
        #     parameters_0_to_1 = experiment_dir / "parameters_0_to_1.json"
        #     real_parameters = experiment_dir / "real_parameters.json"
        #     hyperparameters = experiment_dir / "hyperparameters.json"

        #     with open(parameters_0_to_1, "w") as f:
        #         json.dump(list(results["params"]), f)

        #     with open(real_parameters, "w") as f:
        #         real_params = self.extrapolate_params(results["params"])
            #     real_x = real_params[::3]
            #     real_y = real_params[1::3]
            #     real_coordinates_list = [[real_x[i], real_y[i], 0] for i in range(len(real_x))]
            #     real_radiuses = real_params[2::3]

            #     real_coordinates_and_radiuses = {
            #         "coordinates": list(real_coordinates_list),
            #         "radiuses": list(real_radiuses)
            #     }

            #     json.dump(real_coordinates_and_radiuses, f)

            # with open(hyperparameters, "w") as f:
            #     # Преобразуем объект Sphere в словарь с его параметрами
            #     if isinstance(self.object_to_mimic, list):
            #         object_params = [{
            #             "radius": sphere.radius,
            #             "position": sphere.position,
            #             "refractive_index": {
            #                 "real": sphere.refractive_index.real,
            #                 "imag": sphere.refractive_index.imag
            #             }
            #         } for sphere in self.object_to_mimic]
            #     else:
            #         object_params = {
            #             "radius": self.object_to_mimic.radius,
            #             "position": self.object_to_mimic.position,
            #             "refractive_index": {
            #                 "real": self.object_to_mimic.refractive_index.real,
            #                 "imag": self.object_to_mimic.refractive_index.imag
            #             }
            #         }

            #     hyperparameters = {
            #         "object_to_mimic": object_params,
            #         "vacuum_wavelength": self.vacuum_wavelength,
            #         "angeles_to_mimic": list(self.angeles_to_mimic),  # преобразуем numpy array в list
            #         "side_length": self.side_length,
            #         "number_of_cells": self.number_of_cells,
            #         "refractive_index": self.refractive_index,
            #         "iterations": self.iterations,
            #         "seed": self.seed
            #     }

            #     json.dump(hyperparameters, f)
            # return results 
        
        def plot_optimized_structure(results):
            real_params = self.extrapolate_params(results["params"])
            real_x = real_params[::3]
            real_y = real_params[1::3]
            real_coordinates_list = [[real_x[i], real_y[i], 0] for i in range(len(real_x))]
            real_radiuses = real_params[2::3]
            
            surface = Sphere_surface(self.number_of_cells, self.side_length, self.refractive_index)
            surface.__spheres_add__(coordinates_list=real_coordinates_list, spheres_radius_list=real_radiuses)
            surface.mesh_generation()

            surface.spheres_plot(save_path=experiment_dir / 'spheres_surface_projection.pdf')

        def plot_progress(results):
            fig = plt.figure(figsize=(10, 10))
            generations = range(len(results["progress"])) 
            plt.plot(generations, results["progress"])
            plt.title('Optimization progress')
            plt.xlabel('Generation')
            plt.ylabel('Loss value')
            plt.yscale('log')
            plt.savefig(experiment_dir / 'Optimization_progress.pdf')
            plt.close()

        def plot_spectrum(results):
            real_params = self.extrapolate_params(results["params"])
            real_x = real_params[::3]
            real_y = real_params[1::3]
            real_coordinates_list = [[real_x[i], real_y[i], 0] for i in range(len(real_x))]
            real_radiuses = real_params[2::3]
            
            surface = Sphere_surface(self.number_of_cells, self.side_length, self.refractive_index)
            surface.__spheres_add__(coordinates_list=real_coordinates_list, spheres_radius_list=real_radiuses)

            fig = plt.figure(figsize=(10, 10))
            whole_dscs_surface, whole_dscs_object = calculate_spectrum(surface, self.object_to_mimic, self.vacuum_wavelength)
            surface_array = [np.arange(0, 360, 1), whole_dscs_surface]
            object_array = [np.arange(0, 360, 1), whole_dscs_object]
            plt.plot(*surface_array, label='surface', linewidth=2)
            plt.plot(*object_array, label='object', linewidth=2)
            
            
            angles_indices = np.round(np.degrees(self.angeles_to_mimic)).astype(int)
            
            target_values_object = whole_dscs_object[angles_indices]
            target_values_surface = whole_dscs_surface[angles_indices]
            
            plt.scatter(angles_indices, target_values_object, color='red', s=100, label='target angles object')
            plt.scatter(angles_indices, target_values_surface, color='blue', s=100, label='target angles surface')
            
            plt.legend()
            plt.xlabel('Angle')
            plt.ylabel('DSCS')
            plt.title('Spectrum')
            plt.grid()
            plt.yscale('log')
            plt.savefig(experiment_dir / 'spectrum.pdf')
            plt.close()


        
        return plot_optimized_structure(results), plot_progress(results), plot_spectrum(results)
    






if __name__ == "__main__":
    # object_to_mimic = [particles.FiniteCylinder(position=[0, 0, 250],  # Подняли вверх
    #                                        refractive_index=2, 
    #                                        cylinder_radius=15,
    #                                        cylinder_height=50,
    #                                        euler_angles=[0, 0, 0])]
    
    object_to_mimic = Sphere_surface(
        number_of_cells=2,
        side_length=3.0,
        reflective_index=4
    )
    
    object_to_mimic.mesh_generation()
    
    coordinates_list = [
        [0.75, 0.75, 0],  
        [1.5, 1.5, 0],  
        [0.75, 1.5, 0],  
        [1.5, 0.75, 0],  
    ]
    
    spheres_radius_list = [0.2] * 4
    
    object_to_mimic.__spheres_add__(spheres_radius_list, coordinates_list)

    optimizer = Optimization(object_to_mimic=object_to_mimic, 
                        vacuum_wavelength=0.5, 
                        angeles_to_mimic=np.array([np.deg2rad(30), np.deg2rad(150), np.deg2rad(210), np.deg2rad(280)]), 
                        side_length=3.0, 
                        number_of_cells=2, 
                        refractive_index=4, 
                        iterations=500, 
                        seed=44
                        )

    optimized_surface = optimizer.optimize()
    
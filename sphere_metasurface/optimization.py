import numpy as np
import random
import cmaes
from tqdm import tqdm

import smuthi.initial_field as initial_field
import smuthi.particles as particles

from parametrization import Sphere_surface
from fitness_function import calculate_loss

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

    
    def optimize(self):
        """
        Optimize the sphere surface using CMA-ES algorithm.
        """
        self.surface.mesh_generation()

        n_spheres = self.number_of_cells**2
        random.seed(self.seed)
        initial_params = [random.uniform(0, 1) for _ in range(3*n_spheres)]
        
        cell_size = self.surface.side_length / self.surface.number_of_cells

        def extrapolate_params(params):
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


        def objective_function(params):
          
            real_params = extrapolate_params(params)
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

        population_size = 3
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
                print(params)
                value = objective_function(params)

                values.append(value)
                if value < max_value:
                    max_value = value
                    max_params = params
                    cnt += 1

                solutions.append((params, value))
            # print(solutions)
            # print(values)
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
        return results
    







object_to_mimic = [particles.Sphere(radius=0.1, position=[2, 2, 0], refractive_index=complex(5, 2)), 
                   particles.Sphere(radius=0.1, position=[0, 0, 0], refractive_index=complex(5, 2))]

optimizer = Optimization(object_to_mimic=object_to_mimic, 
                        vacuum_wavelength=0.5, 
                        angeles_to_mimic=np.array([np.pi/3, np.pi/4, np.pi/6]), 
                        side_length=3.0, 
                        number_of_cells=3, 
                        refractive_index=4, 
                        iterations=3, 
                        seed=43
                        )

optimized_surface = optimizer.optimize()

print(optimized_surface)
    
# import numpy as np
# import matplotlib.pyplot as plt
# import smuthi.simulation
# import smuthi.layers
import smuthi.particles

# # Определяем параметры слоев
# layer_system = smuthi.layers.LayerSystem(
#     thicknesses=[0, 20, 30, 40, 0],  # Примерные толщины слоев
#     refractive_indices=[1.0, 2.0, 1.75, 1.5, 1.0]  # Показатели преломления
# )

# # Определяем параметры цилиндров
cylinders = [
    # smuthi.particles.FiniteCylinder(position=(0, 100, 10), cylinder_radius=50, refractive_index=2.5, cylinder_height=200),
    smuthi.particles.FiniteCylinder(position=(50, 50, 40), cylinder_radius=30, refractive_index=2.2, cylinder_height=70),
    # smuthi.particles.FiniteCylinder(position=(-100, -50, 20), cylinder_radius=40, refractive_index=1.8, cylinder_height=180)
]

# # Функция для отрисовки структуры
# def plot_structure(cylinders, layer_system):
#     fig, ax = plt.subplots(figsize=(6, 8))
    
#     # Визуализация слоев
#     z_positions = np.cumsum(layer_system.thicknesses)
#     print(layer_system.thicknesses)
#     for i, z in enumerate(z_positions[:-1]):
#         ax.axhline(y=z, color='gray', linestyle='--', label=f'Layer {i}')
    
#     # Визуализация цилиндров
#     for cyl in cylinders:
#         circle = plt.Circle((cyl.position[0], cyl.position[2]), cyl.cylinder_radius, color='b', alpha=0.5)
#         ax.add_patch(circle)
#         ax.plot([cyl.position[0], cyl.position[0]], [cyl.position[2] - cyl.cylinder_height/2, cyl.position[2] + cyl.cylinder_height/2], 'b')
    
#     ax.set_xlabel("X position (nm)")
#     ax.set_ylabel("Z position (nm)")
#     ax.set_title("Projection of Smuthi Structure")
#     ax.set_xlim(-150, 150)
#     ax.set_ylim(0, 200)
#     ax.legend()
#     ax.set_aspect('equal')
#     plt.savefig('123')

# plot_structure(cylinders, layer_system)

# import numpy as np
# import matplotlib.pyplot as plt
# import smuthi.simulation
# import smuthi.layers
# import smuthi.particles
# from matplotlib.patches import Ellipse

# # Определяем параметры слоев
# layer_system = smuthi.layers.LayerSystem(
#     thicknesses=[0, 20, 50, 0],  # Примерные толщины слоев
#     refractive_indices=[1.0, 1.5, 2.0, 1.0]  # Показатели преломления
# )

# # Определяем параметры цилиндров (добавляем угол наклона)
# cylinders = [
#     {"position": (0, 0, 10), "radius": 5, "refractive_index": 2.5, "height": 20, "tilt": 0},
#     {"position": (10, 10, 30), "radius": 3, "refractive_index": 2.2, "height": 15, "tilt": 20},
#     {"position": (-10, -5, 20), "radius": 4, "refractive_index": 1.8, "height": 18, "tilt": 45}
# ]

# # Функция для отрисовки структуры с учетом наклона цилиндров
# def plot_structure(cylinders, layer_system):
#     fig, ax = plt.subplots(figsize=(6, 8))
    
#     # Визуализация слоев
#     z_positions = np.cumsum(layer_system.thicknesses)
#     for i, z in enumerate(z_positions[:-1]):
#         ax.axhline(y=z, color='gray', linestyle='--', label=f'Layer {i}')
    
#     # Визуализация цилиндров
#     for cyl in cylinders:
#         tilt = cyl["tilt"]
#         aspect_ratio = np.cos(np.radians(tilt))  # Проекция окружности в эллипс
#         ellipse = Ellipse(
#             (cyl["position"][0], cyl["position"][2]), 
#             width=2 * cyl["radius"], height=2 * cyl["radius"] * aspect_ratio,
#             angle=0, color='b', alpha=0.5
#         )
#         ax.add_patch(ellipse)
#         ax.plot([cyl["position"][0], cyl["position"][0]],
#                 [cyl["position"][2] - cyl["height"]/2, cyl["position"][2] + cyl["height"]/2], 'b')
    
#     ax.set_xlabel("X position (nm)")
#     ax.set_ylabel("Z position (nm)")
#     ax.set_title("Projection of Smuthi Structure with Tilted Cylinders")
#     ax.set_xlim(-50, 50)
#     ax.set_ylim(0, max(z_positions))
#     ax.legend()
#     ax.set_aspect('equal')
#     plt.savefig('456')

# plot_structure(cylinders, layer_system)

# import numpy as np
# import matplotlib.pyplot as plt
# import smuthi.simulation
# import smuthi.layers
# import smuthi.particles
# from matplotlib.patches import Ellipse

# # Определяем параметры слоев
# layer_system = smuthi.layers.LayerSystem(
#     thicknesses=[0, 200, 500],  # Примерные толщины слоев
#     refractive_indices=[1.0, 1.5, 2.0]  # Показатели преломления
# )

# # Определяем параметры цилиндров (добавляем угол наклона)
# cylinders = [
#     {"position": (0, 0, 100), "radius": 50, "refractive_index": 2.5, "height": 400, "tilt": 45},
#     {"position": (100, 100, 300), "radius": 30, "refractive_index": 2.2, "height": 150, "tilt": 20},
#     {"position": (-100, -50, 200), "radius": 40, "refractive_index": 1.8, "height": 180, "tilt": 45}
# ]

# # Функция для отрисовки структуры с учетом наклона цилиндров
# def plot_structure(cylinders, layer_system):
#     fig, ax = plt.subplots(figsize=(6, 8))
    
#     # Визуализация слоев
#     z_positions = np.cumsum(layer_system.thicknesses)
#     for i, z in enumerate(z_positions[:-1]):
#         ax.axhline(y=z, color='gray', linestyle='--', label=f'Layer {i}')
    
#     # Визуализация цилиндров
#     for cyl in cylinders:
#         tilt = cyl["tilt"]
#         aspect_ratio = np.cos(np.radians(tilt))  # Проекция окружности в эллипс
#         ellipse = Ellipse(
#             (cyl["position"][0], cyl["position"][2]), 
#             width=2 * cyl["radius"], height=2 * cyl["radius"] * aspect_ratio,
#             angle=0, color='b', alpha=0.5
#         )
#         ax.add_patch(ellipse)
        
#         # Рассчитываем проекцию высоты цилиндра с учетом наклона
#         projected_height = cyl["height"] * np.cos(np.radians(tilt))
#         ax.plot([cyl["position"][0], cyl["position"][0]],
#                 [cyl["position"][2] - projected_height / 2, cyl["position"][2] + projected_height / 2], 'b')
    
#     ax.set_xlabel("X position (nm)")
#     ax.set_ylabel("Z position (nm)")
#     ax.set_title("Projection of Smuthi Structure with Tilted Cylinders")
#     ax.set_xlim(-150, 150)
#     ax.set_ylim(0, max(z_positions))
#     ax.legend()
#     ax.set_aspect('equal')
#     plt.savefig('789')

# plot_structure(cylinders, layer_system)

# import numpy as np
# import matplotlib.pyplot as plt
# import smuthi.simulation
# import smuthi.layers
# import smuthi.particles
# from matplotlib.patches import Ellipse

# # Определяем параметры слоев
# layer_system = smuthi.layers.LayerSystem(
#     thicknesses=[0, 200, 500],  # Примерные толщины слоев
#     refractive_indices=[1.0, 1.5, 2.0]  # Показатели преломления
# )

# # Определяем параметры цилиндров (добавляем угол наклона относительно оси OX)
# cylinders = [
#     {"position": (0, 0, 100), "radius": 50, "refractive_index": 2.5, "height": 150, "tilt": 40},
#     {"position": (100, 100, 300), "radius": 30, "refractive_index": 2.2, "height": 150, "tilt": 20},
#     {"position": (-100, -50, 200), "radius": 40, "refractive_index": 1.8, "height": 180, "tilt": 45}
# ]

# # Функция для отрисовки структуры с учетом наклона цилиндров относительно оси OX
# def plot_structure(cylinders, layer_system):
#     fig, ax = plt.subplots(figsize=(6, 8))
    
#     # Визуализация слоев
#     z_positions = np.cumsum(layer_system.thicknesses)
#     for i, z in enumerate(z_positions[:-1]):
#         ax.axhline(y=z, color='gray', linestyle='--', label=f'Layer {i}')
    
#     # Визуализация цилиндров
#     for cyl in cylinders:
#         tilt = cyl["tilt"]
#         aspect_ratio = np.sin(np.radians(tilt))  # Проекция окружности в эллипс относительно OX
#         ellipse = Ellipse(
#             (cyl["position"][0], cyl["position"][2]), 
#             width=2 * cyl["radius"], height=2 * cyl["radius"] * aspect_ratio,
#             angle=0, color='b', alpha=0.5
#         )
#         ax.add_patch(ellipse)
    
#     ax.set_xlabel("X position (nm)")
#     ax.set_ylabel("Z position (nm)")
#     ax.set_title("Projection of Smuthi Structure with Tilted Cylinders (Tilt Relative to OX)")
#     ax.set_xlim(-150, 150)
#     ax.set_ylim(0, max(z_positions))
#     ax.legend()
#     ax.set_aspect('equal')
#     plt.savefig('789')

# plot_structure(cylinders, layer_system)

import numpy as np
import plotly.graph_objects as go
import smuthi.simulation
import smuthi.layers
import smuthi.particles

# Определяем параметры слоев
layer_system = smuthi.layers.LayerSystem(
    thicknesses=[0, 200, 500],  # Примерные толщины слоев
    refractive_indices=[1.0, 1.5, 2.0]  # Показатели преломления
)

# Определяем параметры цилиндров (добавляем угол наклона относительно оси OX)
cylinders = [
    {"position": (0, 0, 100), "radius": 50, "refractive_index": 2.5, "height": 200, "tilt": 45}
]

# Функция для построения цилиндра в 3D
def create_cylinder(x, y, z, radius, height, tilt, resolution=20):
    theta = np.linspace(0, 2 * np.pi, resolution)
    z_base = np.linspace(z - height / 2, z + height / 2, 2)
    theta_grid, z_grid = np.meshgrid(theta, z_base)
    
    x_grid = radius * np.cos(theta_grid) + x
    y_grid = radius * np.sin(theta_grid) + y * np.cos(np.radians(tilt))
    z_grid = z_grid + y * np.sin(np.radians(tilt))
    
    return x_grid, y_grid, z_grid

# Создаем графический объект
fig = go.Figure()

# Визуализация слоев
z_positions = np.cumsum(layer_system.thicknesses)
for z in z_positions[:-1]:
    fig.add_trace(go.Mesh3d(
        x=[-200, 200, 200, -200],
        y=[-200, -200, 200, 200],
        z=[z, z, z, z],
        color='gray', opacity=0.5, name=f'Layer at {z} nm'
    ))

# Визуализация цилиндров
for cyl in cylinders:
    x_grid, y_grid, z_grid = create_cylinder(
        cyl["position"][0], cyl["position"][1], cyl["position"][2],
        cyl["radius"], cyl["height"], cyl["tilt"]
    )
    fig.add_trace(go.Surface(x=x_grid, y=y_grid, z=z_grid, colorscale='Blues', opacity=1))

# Настройки графика
fig.update_layout(
    scene=dict(
        xaxis_title='X position (nm)',
        yaxis_title='Y position (nm)',
        zaxis_title='Z position (nm)'
    ),
    title="3D Visualization of Smuthi Structure with Tilted Cylinders"
)

fig.show()

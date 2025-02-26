import plotly.graph_objects as go
import numpy as np
import smuthi.particles

# def plot_cylinder(position, radius, height, orientation, color='blue'):
#     # Генерация точек цилиндра (упрощенный пример)
#     z = np.linspace(0, height, 50)
#     theta = np.linspace(0, 2*np.pi, 50)
#     theta_grid, z_grid = np.meshgrid(theta, z)
#     x_grid = radius * np.cos(theta_grid)
#     y_grid = radius * np.sin(theta_grid)
    
#     # Учет ориентации и положения
#     # (здесь требуется применение матрицы поворота, зависит от задания ориентации)
#     # Пример для ориентации вдоль оси Z:
#     x = x_grid + position[0]
#     y = y_grid + position[1]
#     z = z_grid + position[2]

#     return go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]])

# fig = go.Figure()

cylinder = smuthi.particles.FiniteCylinder(position=[0, 0, 0],
                                  refractive_index=1+6j,
                                  cylinder_radius=50,
                                  cylinder_height=150,
                                  l_max=4,
                                  euler_angles=[0,0,0])


# # List of all scattering particles
# particles = [cylinder]

# for particle in particles:
#     if isinstance(particle, smuthi.particles.FiniteCylinder):  # Если тип частицы - цилиндр
#         fig.add_trace(plot_cylinder(particle.position,  
#                                     particle.cylinder_height,
#                                     particle.cylinder_radius,
#                                     particle.euler_angles))
# fig.show()

# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, Ellipse

# def plot_cylinder_projection(ax, position, radius, height, angle=0, color='blue'):
#     """Визуализация проекции цилиндра на плоскость XZ.
    
#     Args:
#         ax (matplotlib.axes.Axes): Оси для отрисовки
#         position (list): [x, z] координаты центра цилиндра
#         radius (float): Радиус основания
#         height (float): Высота цилиндра
#         angle (float): Угол поворота вокруг оси Y (в градусах)
#         color (str): Цвет заливки
#     """
#     # Угол в радианах
#     theta = np.radians(angle)
    
#     # Параметры прямоугольника (основная часть)
#     rect_width = 2 * radius
#     rect_height = height
    
#     # Смещение из-за поворота
#     dx = height/2 * np.sin(theta)
#     dz = height/2 * np.cos(theta)
    
#     # Позиция нижнего левого угла прямоугольника
#     rect_x = position[0] - radius - dx
#     rect_z = position[1] - dz
    
#     # Прямоугольник (боковая поверхность)
#     rect = Rectangle((rect_x, rect_z), 
#                      width=rect_width, 
#                      height=rect_height,
#                      angle=angle,
#                      color=color, 
#                      alpha=0.5)
#     ax.add_patch(rect)
    
#     # # Эллипсы для торцов (проекция окружностей)
#     # for sign in [-1, 1]:
#     #     # Центр эллипса
#     #     cx = position[0] + sign * (height/2) * np.sin(theta)
#     #     cz = position[1] + sign * (height/2) * np.cos(theta)
        
#     #     # Эллипс (ширина зависит от угла)
#     #     ellipse = Ellipse((cx, cz), 
#     #                       width=2*radius, 
#     #                       height=2*radius*np.cos(theta),
#     #                       angle=angle,
#     #                       color=color, 
#     #                       alpha=0.5)
#     #     ax.add_patch(ellipse)

# # Пример использования
# fig, ax = plt.subplots(figsize=(8, 6))

# # Параметры цилиндра
# position = [0, 0]   # x, z (предполагаем проекцию на XZ-плоскость)
# radius = 1.0        # Радиус
# height = 3.0        # Высота
# angle = 30          # Угол поворота относительно оси Y

# # Рисуем цилиндр
# plot_cylinder_projection(ax, position, radius, height, angle, color='blue')

# # Настройка осей
# ax.set_xlim(-3, 3)
# ax.set_ylim(-2, 4)
# ax.set_aspect('equal')
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# ax.grid(True)
# plt.title('Проекция цилиндра на плоскость XZ')
# plt.savefig('projections')

from scipy.spatial.transform import Rotation
import numpy as np
import plotly.graph_objects as go

def plot_3d_cylinder(position, direction, radius, height, color='blue'):
    """Создает 3D-цилиндр с заданными параметрами.
    
    Args:
        position (list): [x, y, z] координаты центра цилиндра
        direction (list): Вектор направления [dx, dy, dz] (ось цилиндра)
        radius (float): Радиус
        height (float): Высота
        color (str): Цвет
    """
    # Нормализация вектора направления
    direction = np.array(direction) / np.linalg.norm(direction)
    
    # Углы для параметризации окружности
    theta = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, height, 2)
    
    # Параметризация цилиндра
    t, h = np.meshgrid(theta, v)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = h - height / 2  # Центрирование по высоте
    
    # Преобразование в массив векторов формы (P, 3)
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    
    # Поворот и перемещение
    axis = np.array([0, 0, 1])  # Исходное направление (ось Z)
    target_axis = direction
    rot_axis = np.cross(axis, target_axis)
    rot_angle = np.arccos(np.dot(axis, target_axis))
    
    # Применение поворота
    rotation = Rotation.from_rotvec(rot_axis * rot_angle)
    points_rotated = rotation.apply(points)  # Теперь форма (P, 3)
    
    # Перенос в позицию
    x_final = points_rotated[:, 0] + position[0]
    y_final = points_rotated[:, 1] + position[1]
    z_final = points_rotated[:, 2] + position[2]
    
    # Возвращаем Surface для Plotly
    return go.Surface(
        x=x_final.reshape(x.shape),
        y=y_final.reshape(y.shape),
        z=z_final.reshape(z.shape),
        colorscale=[[0, color], [1, color]]
    )

# Пример использования
fig = go.Figure()

# Цилиндр 1 (вертикальный)
fig.add_trace(plot_3d_cylinder(
    position=[0, 0, 0],
    direction=[0, 0, 1],
    radius=1.0,
    height=3.0,
    color='blue'
))

# Цилиндр 2 (наклонный)
fig.add_trace(plot_3d_cylinder(
    position=[2, 0, 0],
    direction=[1, 1, 1],
    radius=0.8,
    height=4.0,
    color='red'
))

fig.update_layout(
    scene=dict(
        aspectmode='data',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    ),
    title='3D Цилиндры'
)
fig.show()
import porespy as ps
import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.color import rgba2rgb

# Параметры
voxel_size = 1e-6  # Размер вокселя в метрах (1 мкм)

# Шаг 1: Загрузка и обработка изображения
im = io.imread('krugi3 1.png')
print(im.shape)

# Если изображение цветное, преобразуем его в оттенки серого
if len(im.shape) == 3:
    if im.shape[2] == 4:
        im_rgb = rgba2rgb(im)
        im_gray = color.rgb2gray(im_rgb)
    else:
        im_gray = color.rgb2gray(im)
else:
    im_gray = im.copy()

# Преобразование в бинарное изображение (пористая структура)
thresh = filters.threshold_otsu(im_gray)
im_binary = im_gray > thresh

# Инвертирование бинарного изображения
im_binary = np.logical_not(im_binary)
print(im_binary.shape)

# Визуализация бинарного изображения (пористая структура)
plt.figure(figsize=(12, 6))
plt.imshow(im_binary, cmap='gray')
plt.title('Инвертированное пористое изображение')
plt.axis('off')
plt.show()

# Расчет пористости изображения
calculated_porosity = np.sum(im_binary) / im_binary.size
print(f"Пористость изображения: {calculated_porosity:.4f}")

# Шаг 2: Преобразование изображения в сеть пор с использованием SNOW2
snow = ps.networks.snow2(
    im_binary,
    voxel_size=voxel_size)

# Импорт сети в OpenPNM
pn = op.io.network_from_porespy(snow.network)

# Назначение геометрических свойств пор и горловин
if 'pore.inscribed_diameter' in snow.network.keys():
    pn['pore.diameter'] = snow.network['pore.inscribed_diameter']
else:
    print("Ключ 'pore.inscribed_diameter' не найден в snow.network.")

pn.add_model(propname='throat.diameter',
             model=op.models.geometry.throat_size.from_neighbor_pores,
             mode='min')

pn.add_model(propname='throat.length',
             model=op.models.geometry.throat_length.spheres_and_cylinders)

pn.add_model(propname='throat.cross_sectional_area',
             model=op.models.geometry.throat_cross_sectional_area.cylinder)

pn.add_model(propname='throat.hydraulic_size_factors',
             model=op.models.geometry.hydraulic_size_factors.spheres_and_cylinders)

pn.regenerate_models()

# Проверка корректности сети и удаление изолированных пор
h = op.utils.check_network_health(pn)
if len(h['disconnected_pores']) > 0:
    op.topotools.trim(network=pn, pores=h['disconnected_pores'])
    pn.regenerate_models()
else:
    print("Изолированных пор не обнаружено.")

# Определение граничных пор для задания граничных условий
tol = voxel_size / 2
y_coords = pn['pore.coords'][:, 1]
top_pores = pn.Ps[y_coords >= y_coords.max() - tol]
bottom_pores = pn.Ps[y_coords <= y_coords.min() + tol]

# Создание фазовой модели (вода) и добавление физических моделей
water = op.phase.Water(network=pn)
water['pore.viscosity'] = 0.001  # Вязкость воды при 20°C в Па·с

water.add_model(propname='throat.hydraulic_conductance',
                model=op.models.physics.hydraulic_conductance.hagen_poiseuille)

water.regenerate_models()

# Моделирование потока
flow = op.algorithms.StokesFlow(network=pn, phase=water)
flow.set_value_BC(pores=top_pores, values=1000000)  # Давление в Па
flow.set_value_BC(pores=bottom_pores, values=0)
flow.run()

# Расчет проницаемости
Q_total = np.sum(np.abs(flow.rate(pores=top_pores))) 
A = im_binary.shape[1] * 1e-12 * 5
L = im_binary.shape[0] * 1e-6
mu = 0.001                                       # Вязкость, Па·с
dP = 1000000                                      # Разница давлений, Па
K = (Q_total * mu * L) / (A * dP)
darcy_constant = 9.869233e-13
K_darcy = K / darcy_constant * 1e3  # Проницаемость в Дарси (1 Дарси ≈ 9.869e-13 м^2)

print(f"Пористость изображения: {calculated_porosity:.4f}")
print(f"Рассчитанная проницаемость: {K_darcy} мДарси")
print(f"Общий объемный расход: {Q_total:.4e} м³/с")

# Визуализация распределения давления на поровой сети
coords = pn['pore.coords']
conns = pn['throat.conns']

x = coords[:, 1] / voxel_size  # Координата Y в пикселях
y = coords[:, 0] / voxel_size  # Координата X в пикселях

pore_pressures = flow['pore.pressure']
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(im_binary, cmap='gray')

for throat in conns:
    p1, p2 = throat
    x1, y1 = x[p1], y[p1]
    x2, y2 = x[p2], y[p2]
    ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5)

scatter = ax.scatter(x, y, c=pore_pressures, cmap='jet', s=20)
ax.set_title('Распределение давления в порах')
ax.axis('off')
plt.gca().invert_yaxis()
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Давление (Па)')
plt.show()


# Проверка корректности сети без удаления изолированных пор
h = op.utils.check_network_health(pn)
print(f"Количество изолированных пор: {len(h['disconnected_pores'])}")
print(f"Всего пор в сети: {pn.Np}")

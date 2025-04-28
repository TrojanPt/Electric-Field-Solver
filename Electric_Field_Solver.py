'''
利用PyTorch神经网络内嵌拉普拉斯方程 ∇²φ = 0 实现PINN(Physics-Informed Neural Networks)求解三维静电场问题。

requirements:
    torch==2.6.0+cu126
    numpy==2.1.2
    matplotlib==3.10.1
其它版本注意兼容性。

'''

from matplotlib import pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch.nn as nn


# 检查CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置物理常数
epsilon_0 = 1.0


class ElectricPotentialNN(nn.Module):
    def __init__(self, input_dim = 3, hidden_layers = [64, 64, 64, 64]):
        '''
        初始化电势神经网络模型
        Args:
            input_dim(int, default = 3): 输入维度
            hidden_layers(list, default = [64, 64, 64, 64]): 隐藏层神经元数量
        '''
        super().__init__()

        layers = []

        pre_layer_dim = input_dim
        for neurons in hidden_layers:
            layers.append(nn.Linear(pre_layer_dim, neurons))
            layers.append(nn.SiLU())
            pre_layer_dim = neurons

        output_dim = 1
        layers.append(nn.Linear(pre_layer_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, r) -> torch.Tensor:
        '''
        向前传播
        Args:
            r: 输入张量 (N, input_dim)
        Returns:
            torch.Tensor: 输出张量 (N, 1)
        '''
        return self.network(r)
        
    def laplacian(self, r):
        """
        计算电势的拉普拉斯算子值
        Args:
            r: 输入张量 (N, input_dim)
        Returns:
            torch.Tensor: 拉普拉斯算子值 ∇²φ (N, 1)
        """
        r.requires_grad_(True)
        
        # 电势 φ
        phi = self.forward(r)
        
        # 记录 φ 的形状
        ones = torch.ones_like(phi)
        
        # 一阶导数 ∇φ
        dphi_dr = torch.autograd.grad(
            phi, r, 
            grad_outputs=ones, 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        # 二阶导数 ∇²φ = ∆φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
        laplacian = 0
        for i in range(r.shape[1]):
            d2phi_dri2 = torch.autograd.grad(
                dphi_dr[:, i], r, 
                grad_outputs=torch.ones_like(dphi_dr[:, i]),
                create_graph=True,
                retain_graph=True
            )[0][:, i]
            laplacian += d2phi_dri2
        
        return laplacian.unsqueeze(1)

    def compute_electric_field(self, r):
        """
        计算r处的电场（电势的负梯度）
        
        Args:
            r: 坐标
        
        Returns:
            torch.Tensor: 电场向量
        """
        r.requires_grad_(True)
        
        # 电势 φ
        phi = self.forward(r)
        
        # ∇φ = (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)
        grad_outputs = torch.ones_like(phi)
        gradient = torch.autograd.grad(
            phi, r, 
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # E = -∇φ
        electric_field = -gradient
        
        return electric_field


class Field:
    def __init__(self, dim=3):
        self.dim = dim
        self.electric_field = [0.0] * dim

        self.conductors = []  # 存储所有导体
        self.add_electrical_conductor = self.ConductorAdder(self)

        self.charged_bodies = []  # 存储所有带电体
        self.add_charged_body = self.ChargedBodyAdder(self)
    
    class ConductorAdder:
        def __init__(self, field):
            self.field = field
        
        def cylinder(self, center: tuple, radius: float, height: float, is_grounded: bool = True):
            """
            添加圆柱体导体
            
            Args:
                center: 圆柱体中心坐标，形如 (x, y, z)
                radius: 圆柱体半径
                height: 圆柱体高度
                is_grounded(bool, default = True): 导体是否接地
            """
            self.field.conductors.append({
                "shape": "cylinder",
                "center": center,
                "radius": radius,
                "height": height,
                "is_grounded": is_grounded,
                "potential": 0.0    # 初始电势，对孤立导体，参数会在训练过程中更新
            })
            return self.field
        
        def sphere(self, center: tuple, radius, is_grounded: bool = True):
            """
            添加球形导体
            
            Args:
                center: 球体中心坐标，形如 (x, y, z)
                radius: 球体半径
                is_grounded(bool, default = True): 导体是否接地
            """
            self.field.conductors.append({
                "shape": "sphere",
                "center": center,
                "radius": radius,
                "is_grounded": is_grounded,
                "potential": 0.0    # 初始电势
            })
            return self.field
        
        def cube(self, center: tuple, a, b, c, is_grounded: bool = True):
            """
            添加立方体导体
            
            Args:
                center: 立方体中心坐标，形如 (x, y, z)
                a, b, c: 立方体的长宽高
                is_grounded(bool, default = True): 导体是否接地
            """
            self.field.conductors.append({
                "shape": "cube",
                "center": center,
                "dimensions": (a, b, c),
                "is_grounded": is_grounded,
                "potential": 0.0    # 初始电势
            })
            return self.field
    
    class ChargedBodyAdder:
        def __init__(self, field):
            self.field = field
        
        def sphere(self, center: tuple, radius: float, density: float):
            """
            添加带电球体
            
            Args:
                center: 球体中心坐标，形如 (x, y, z)
                radius: 球体半径
                density: 球体的电荷密度
            """
            self.field.charged_bodies.append({
                "shape": "sphere",
                "center": center,
                "radius": radius,
                "density": density,
            })
            return self.field

    def add_electric_field(self, E_x, E_y, E_z):
        """
        添加匀强电场
        """
        self.electric_field = [E_x, E_y, E_z]
    
    def to_potential(self, point):
        """
        将电场转换为电势
        """
        x, y, z = point
        E_x, E_y, E_z = self.electric_field
        
        # 匀强电场，电势 φ = -E * r
        phi = -(E_x * x + E_y * y + E_z * z)
        
        return phi

    def is_inside_conductor(self, point) -> bool:
        """
        判断点是否在任何导体内
        """
        x, y, z = point
        
        for conductor in self.conductors:
            if conductor["shape"] == "cylinder":
                # 圆柱体判断
                cx, cy, cz = conductor["center"]
                r = conductor["radius"]
                h = conductor["height"]
                
                # 计算点到圆柱轴线的距离
                distance_to_axis = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                
                # 判断点是否在圆柱体高度范围内
                half_height = h / 2
                if distance_to_axis <= r and cz - half_height <= z <= cz + half_height:
                    return True
                    
            elif conductor["shape"] == "sphere":
                # 球体判断
                cx, cy, cz = conductor["center"]
                r = conductor["radius"]
                
                # 计算点到球心的距离
                distance = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) ** 0.5
                
                if distance <= r:
                    return True
                    
            elif conductor["shape"] == "cube":
                # 立方体判断
                cx, cy, cz = conductor["center"]
                a, b, c = conductor["dimensions"]
                
                # 计算立方体的边界
                half_a, half_b, half_c = a/2, b/2, c/2
                
                if (cx - half_a <= x <= cx + half_a and 
                    cy - half_b <= y <= cy + half_b and 
                    cz - half_c <= z <= cz + half_c):
                    return True
        
        return False
    
    def is_inside_charged_body(self, point) -> float:
        """
        判断点是否在任何带电体内
        Args:
            point(tuple): 点坐标 (x, y, z)

        Returns:
            float: 如果在带电体内，返回电荷密度，否则返回0.0
        """
        x, y, z = point
        
        for charged_body in self.charged_bodies:
            if charged_body["shape"] == "sphere":
                # 球体判断
                cx, cy, cz = charged_body["center"]
                r = charged_body["radius"]
                
                # 计算点到球心的距离
                distance = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) ** 0.5
                
                if distance <= r:
                    return charged_body["density"]
        
        return 0.0

    # not used in the current code
    def is_nearby_conductor(self, point, rel_tolerance = 0.01) -> bool:
        """
        判断点是否在任何导体附近

        Args:
            point: 点坐标
            rel_tolerance: 相对导体线度的容差范围，导体线度 l 可取为该导体形状的入参的几何平均，例如cube则取 l = (a*b*c)^(1/3)，则 tolerance = rel_tolerance * l

        Returns:
            bool: 点到导体的距离是否小于tolerance，如果是则返回True，否则返回False
        """
        x, y, z = point
    
        for conductor in self.conductors:
            if conductor["shape"] == "cylinder":
                # 圆柱体
                cx, cy, cz = conductor["center"]
                r = conductor["radius"]
                h = conductor["height"]
                
                # 导体线度，取几何平均
                l = (2 * r * h) ** (1/2)
                tolerance = rel_tolerance * l
                
                # 计算点到圆柱轴线的距离
                distance_to_axis = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                
                # 计算点到圆柱底面或顶面的距离
                half_height = h / 2
                z_distance = 0
                if z > cz + half_height:
                    z_distance = z - (cz + half_height)
                elif z < cz - half_height:
                    z_distance = (cz - half_height) - z
                
                # 如果点在圆柱体上方或下方
                if z_distance > 0:
                    # 如果点在轴线正上方或下方
                    if distance_to_axis <= r:
                        if z_distance < tolerance:
                            return True
                    # 如果点在圆柱边缘上方或下方
                    else:
                        edge_distance = ((distance_to_axis - r) ** 2 + z_distance ** 2) ** 0.5
                        if edge_distance < tolerance:
                            return True
                # 如果点在圆柱体高度范围内，检查到侧面的距离
                else:
                    if abs(distance_to_axis - r) < tolerance:
                        return True
                    
            elif conductor["shape"] == "sphere":
                # 球体
                cx, cy, cz = conductor["center"]
                r = conductor["radius"]
                
                # 球体线度就是直径
                l = 2 * r
                tolerance = rel_tolerance * l
                
                # 计算点到球心的距离
                distance = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) ** 0.5
                
                # 检查点到球面的距离
                if abs(distance - r) < tolerance:
                    return True
                    
            elif conductor["shape"] == "cube":
                # 立方体
                cx, cy, cz = conductor["center"]
                a, b, c = conductor["dimensions"]
                
                # 计算立方体的线度（几何平均）
                l = (a * b * c) ** (1/3)
                tolerance = rel_tolerance * l
                
                # 计算立方体的边界
                half_a, half_b, half_c = a/2, b/2, c/2
                
                # 计算点到立方体各面的最小距离
                dx = max(0, abs(x - cx) - half_a)
                dy = max(0, abs(y - cy) - half_b)
                dz = max(0, abs(z - cz) - half_c)
                
                # 如果点在立方体内部
                if dx == 0 and dy == 0 and dz == 0:
                    # 计算到各面的距离
                    dist_to_faces = [
                        half_a - abs(x - cx),
                        half_b - abs(y - cy),
                        half_c - abs(z - cz)
                    ]
                    min_dist = min(dist_to_faces)
                    if min_dist < tolerance:
                        return True
                else:
                    # 计算点到立方体的最短距离
                    distance = (dx**2 + dy**2 + dz**2)**0.5
                    if distance < tolerance:
                        return True
        
        return False

    def determine_comp_domain(self, rel_infinite_distance = 2) -> tuple:
        '''
        根据导体/带电体分布确定矩形区域，后续计算将认为该计算区域的边界近似为无穷远处
        
        Args:
            rel_infinite_distance: 相对导体分布与导体线度的无穷远处的距离，导体线度 l 可取为该导体形状的入参的几何平均，例如cube则取 l = (a*b*c)^(1/3)
                                    则任何导体中心到该区域边界的距离都大于 rel_infinite_distance * l_max
                                    其中 l_max 为所有导体的线度中的最大值
        
        Returns:
            tuple: 计算区域的边界坐标 (xmin, xmax, ymin, ymax, zmin, zmax)
        '''
        if not self.conductors and not self.charged_bodies:
            # 若无导体和带电体，返回默认计算域
            return (-10, 10, -10, 10, -10, 10)
        
        # 记录所有导体和带电体中心点坐标和线度
        centers = []
        linscales = []
        
        for conductor in self.conductors:
            centers.append(conductor["center"])
            
            if conductor["shape"] == "cylinder":
                r = conductor["radius"]
                h = conductor["height"]
                l = (2 * r * h) ** (1/2)
            elif conductor["shape"] == "sphere":
                r = conductor["radius"]
                l = 2 * r
            elif conductor["shape"] == "cube":
                a, b, c = conductor["dimensions"]
                l = (a * b * c) ** (1/3)
            
            linscales.append(l)

        for charged_body in self.charged_bodies:
            centers.append(charged_body["center"])
            
            if charged_body["shape"] == "sphere":
                r = charged_body["radius"]
                l = 2 * r
            
            linscales.append(l)
        
        # 最大线度
        l_max = max(linscales)
        l_max = max(l_max, 10) 
        
        # 计算所有中心点的坐标范围
        min_x = min(c[0] for c in centers)
        max_x = max(c[0] for c in centers)
        min_y = min(c[1] for c in centers)
        max_y = max(c[1] for c in centers)
        min_z = min(c[2] for c in centers)
        max_z = max(c[2] for c in centers)
        
        # 扩展边界，使所有导体与边界的距离大于 rel_infinite_distance * l_max
        boundary_extension = rel_infinite_distance * l_max
        
        xmin = min_x - boundary_extension
        xmax = max_x + boundary_extension
        ymin = min_y - boundary_extension
        ymax = max_y + boundary_extension
        zmin = min_z - boundary_extension
        zmax = max_z + boundary_extension
        
        return (xmin, xmax, ymin, ymax, zmin, zmax)
    
    def visualize(self):
        """
        导体及带电体分布图
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 确定边界
        min_x, max_x, min_y, max_y, min_z, max_z = self.determine_comp_domain()
        
        # 导体
        for conductor in self.conductors:
            if conductor["shape"] == "cylinder":
                self.visualize_cylinder(ax, conductor)
            elif conductor["shape"] == "sphere":
                self.visualize_sphere(ax, conductor)
            elif conductor["shape"] == "cube":
                self.visualize_cube(ax, conductor)

        # 带电体
        for body in self.charged_bodies:
            if body["shape"] == "sphere":
                self.visualize_sphere(ax, body, is_charged_body=True)
        
        # 设置轴标签和标题
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('The distribution of conductors(b) and charged bodies(r)')
        
        # 设置轴范围
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.set_zlim([min_z, max_z])

        # 显示坐标轴
        ax.quiver(min_x, 0, 0, max_x - min_x, 0, 0, color='black', arrow_length_ratio=0.1, alpha=0.5)
        ax.quiver(0, min_y, 0, 0, max_y - min_y, 0, color='black', arrow_length_ratio=0.1, alpha=0.5)
        ax.quiver(0, 0, min_z, 0, 0, max_z - min_z, color='black', arrow_length_ratio=0.1, alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        # plt.savefig("conductors_and_charged_bodies.png")

    def visualize_cylinder(self, ax: Axes3D, conductor):
        """可视化圆柱体"""
        cx, cy, cz = conductor["center"]
        radius = conductor["radius"]
        height = conductor["height"]
        
        # 圆柱体顶部和底部圆的z坐标
        top_z = cz + height/2
        bottom_z = cz - height/2
        
        # 创建圆的参数方程
        theta = np.linspace(0, 2*np.pi, 100)
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        
        # 绘制顶部和底部圆
        ax.plot(x, y, [top_z]*100, 'b-')
        ax.plot(x, y, [bottom_z]*100, 'b-')
        
        # 绘制连接顶部和底部圆的线段
        for i in range(0, 100, 10):
            ax.plot([x[i], x[i]], [y[i], y[i]], [bottom_z, top_z], 'b-')

    def visualize_sphere(self, ax: Axes3D, body, is_charged_body=False):
        """可视化球体"""
        cx, cy, cz = body["center"]
        radius = body["radius"]
        
        # 用u, v参数化球体表面
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        x = cx + radius * np.outer(np.cos(u), np.sin(v))
        y = cy + radius * np.outer(np.sin(u), np.sin(v))
        z = cz + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, color='b' if not is_charged_body else 'r', alpha=0.5)

    def visualize_cube(self, ax: Axes3D, conductor):
        """可视化立方体"""
        cx, cy, cz = conductor["center"]
        a, b, c = conductor["dimensions"]
        
        # 计算立方体的顶点
        half_a, half_b, half_c = a/2, b/2, c/2
        
        # 立方体的八个顶点坐标
        vertices = [
            [cx - half_a, cy - half_b, cz - half_c],
            [cx + half_a, cy - half_b, cz - half_c],
            [cx + half_a, cy + half_b, cz - half_c],
            [cx - half_a, cy + half_b, cz - half_c],
            [cx - half_a, cy - half_b, cz + half_c],
            [cx + half_a, cy - half_b, cz + half_c],
            [cx + half_a, cy + half_b, cz + half_c],
            [cx - half_a, cy + half_b, cz + half_c]
        ]
        
        # 定义立方体的12条边
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 连接顶面和底面的边
        ]
        
        # 绘制立方体的边
        for edge in edges:
            ax.plot3D(
                [vertices[edge[0]][0], vertices[edge[1]][0]],
                [vertices[edge[0]][1], vertices[edge[1]][1]],
                [vertices[edge[0]][2], vertices[edge[1]][2]],
                'b-'
            )


def set_field() -> Field:
    """
    设置电场和导体
    """
    field = Field()
    
    # 添加匀强电场或带电体
    flag = input("Set up a uniform background electric field or charged bodies? (electric field/charged body): ").strip().lower()
    # 匀强电场
    if flag == "electric field":
        print("Set up uniform electric field( V/m )")
        E_x = input("E_x = ")
        E_y = input("E_y = ")
        E_z = input("E_z = ")
        field.add_electric_field(float(E_x), float(E_y), float(E_z))
    # 带电体
    elif flag == "charged body":
        print("Set up charged body")
        # 添加带电体，目前仅支持球体(sphere)
        while True:
            print("Add charged body, currently only support sphere")
            shape = input("Please input the shape of the charged body (sphere) or 'exit' to quit: ").strip().lower()
            if shape == "exit":
                break

            center = tuple(map(float, input("the center coordinates of the sphere (x, y, z): ").split(',')))
            radius = float(input("the radius of the sphere: "))
            density = float(input("the charge density of the sphere: "))
            field.add_charged_body.sphere(center, radius, density)
    
    # 添加导体，目前仅支持圆柱体(cylinder)、球体(sphere)和立方体(cube)
    while True:
        print("Add electrical conductor, currently only support cylinder, sphere and cube")
        shape = input("Please input the shape of the conductor (cylinder/sphere/cube) or 'exit' to quit: ").strip().lower()
        if shape == "exit":
            break

        # 导体类型
        conductor_type = input("Please select the type of conductor (grounded/isolated): ").strip().lower()
        is_grounded = True if conductor_type == "grounded" else False

        if shape == "cylinder":
            center = tuple(map(float, input("the center coordinates of the cylinder (x, y, z): ").split(',')))
            radius = float(input("the radius of the cylinder: "))
            height = float(input("the height of the cylinder: "))
            field.add_electrical_conductor.cylinder(center, radius, height, is_grounded)
        elif shape == "sphere":
            center = tuple(map(float, input("the center coordinates of the sphere (x, y, z): ").split(',')))
            radius = float(input("the radius of the sphere: "))
            field.add_electrical_conductor.sphere(center, radius, is_grounded)
        elif shape == "cube":
            center = tuple(map(float, input("the center coordinates of the cube (x, y, z): ").split(',')))
            a = float(input("the length of the cube: "))
            b = float(input("the width of the cube: "))
            c = float(input("the height of the cube: "))
            field.add_electrical_conductor.cube(center, a, b, c, is_grounded)
    
    return field


def generate_collocation_points(
        field: Field, 
        n_domain:int = 30000, 
        n_far_boundary:int = 6000, 
        n_boundary_per_conductor:int = 1000, 
        n_density_charged_body: int = 100,
        comp_domain: tuple = (-10, 10, -10, 10, -10, 10)
        ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, bool]], torch.Tensor, list[tuple[torch.Tensor, float]]]:
    """
    生成用于训练的采样点。
    
    Args:
        field(Field): 电场与导体设置
        n_domain: 区域内部采样点数量
        n_far_boundary: 空间极远处的边界采样点数量（若不存在背景电场，则不采样）
        n_boundary_per_conductor: 每个导体的边界采样点(及近边界采样点)数量
        n_density_charged_body: 带电体的采样点数密度
        comp_domain: 计算域范围 (xmin, xmax, ymin, ymax, zmin, zmax)
    
    Returns:
        tuple: (domain_r, far_boundary_r, conductors_boundary_list, conductors_near_boundary_r, charged_bodies_list)

            `domain_r`(torch.Tensor): 区域内部的采样点

            `far_boundary_r`(torch.Tensor | None): 空间极远处的边界采样点

            `conductors_boundary_list`(list): 每个导体的边界点列表，格式为[(points, is_grounded), ...]

            `conductors_near_boundary_r`(torch.Tensor): 每个导体的近边界采样点

            `charged_bodies_list`(list): 每个带电体的采样点列表，格式为[(points, density), ...]
    """
    xmin, xmax, ymin, ymax, zmin, zmax = comp_domain
    
    # 1. 生成区域内部的点 ##################################

    # 创建足够多的点以保证n_domain个点在所有导体外部
    n_extra = int(n_domain * 0.5)  # 额外生成，弥补可能被导体占用的点
    
    x = torch.FloatTensor(n_domain + n_extra, 1).uniform_(xmin, xmax)
    y = torch.FloatTensor(n_domain + n_extra, 1).uniform_(ymin, ymax)
    z = torch.FloatTensor(n_domain + n_extra, 1).uniform_(zmin, zmax)
    
    # 合并坐标
    domain_points = torch.cat([x, y, z], dim=1)
    
    # 过滤出不在导体及带电体内部的点
    valid_points = []
    for i in range(domain_points.shape[0]):
        point = domain_points[i].tolist()
        if not field.is_inside_conductor(point) and field.is_inside_charged_body(point) == 0.0:
            # 如果点不在导体和带电体内部，则添加到有效点列表中
            valid_points.append(domain_points[i])
        if len(valid_points) >= n_domain:
            break
    
    # 如果有效点不足，再次尝试生成
    while len(valid_points) < n_domain:
        x_extra = torch.FloatTensor(n_extra, 1).uniform_(xmin, xmax)
        y_extra = torch.FloatTensor(n_extra, 1).uniform_(ymin, ymax)
        z_extra = torch.FloatTensor(n_extra, 1).uniform_(zmin, zmax)
        
        extra_points = torch.cat([x_extra, y_extra, z_extra], dim=1)
        
        for i in range(extra_points.shape[0]):
            point = extra_points[i].tolist()
            if not field.is_inside_conductor(point) and field.is_inside_charged_body(point) == 0.0:
                valid_points.append(extra_points[i])
            if len(valid_points) >= n_domain:
                break
    
    domain_r = torch.stack(valid_points[:n_domain])
    
    # 2. 若存在背景电场，则生成远处边界上的点 ##################################
    if field.electric_field != [0.0, 0.0, 0.0]:
            
        # 生成六个面上的点，每个面上取 n_far_boundary/6 个点
        points_per_face = n_far_boundary // 6
        
        # xmin面
        x_min = torch.full((points_per_face, 1), xmin)
        y_min_face = torch.FloatTensor(points_per_face, 1).uniform_(ymin, ymax)
        z_min_face = torch.FloatTensor(points_per_face, 1).uniform_(zmin, zmax)
        face1 = torch.cat([x_min, y_min_face, z_min_face], dim=1)
        
        # xmax面
        x_max = torch.full((points_per_face, 1), xmax)
        y_max_face = torch.FloatTensor(points_per_face, 1).uniform_(ymin, ymax)
        z_max_face = torch.FloatTensor(points_per_face, 1).uniform_(zmin, zmax)
        face2 = torch.cat([x_max, y_max_face, z_max_face], dim=1)
        
        # ymin面
        x_y_min = torch.FloatTensor(points_per_face, 1).uniform_(xmin, xmax)
        y_min = torch.full((points_per_face, 1), ymin)
        z_y_min = torch.FloatTensor(points_per_face, 1).uniform_(zmin, zmax)
        face3 = torch.cat([x_y_min, y_min, z_y_min], dim=1)
        
        # ymax面
        x_y_max = torch.FloatTensor(points_per_face, 1).uniform_(xmin, xmax)
        y_max = torch.full((points_per_face, 1), ymax)
        z_y_max = torch.FloatTensor(points_per_face, 1).uniform_(zmin, zmax)
        face4 = torch.cat([x_y_max, y_max, z_y_max], dim=1)
        
        # zmin面
        x_z_min = torch.FloatTensor(points_per_face, 1).uniform_(xmin, xmax)
        y_z_min = torch.FloatTensor(points_per_face, 1).uniform_(ymin, ymax)
        z_min = torch.full((points_per_face, 1), zmin)
        face5 = torch.cat([x_z_min, y_z_min, z_min], dim=1)
        
        # zmax面
        x_z_max = torch.FloatTensor(points_per_face, 1).uniform_(xmin, xmax)
        y_z_max = torch.FloatTensor(points_per_face, 1).uniform_(ymin, ymax)
        z_max = torch.full((points_per_face, 1), zmax)
        face6 = torch.cat([x_z_max, y_z_max, z_max], dim=1)
        
        far_boundary_r = torch.cat([face1, face2, face3, face4, face5, face6], dim=0)
    
    else:
        # 如果没有背景电场，则不生成远处边界点
        far_boundary_r = None
    
    # 3. 为每个导体生成边界点 ##################################

    conductors_boundary_list = []
    conductors_near_boundary_r = []
    
    for conductor in field.conductors:
        if conductor["shape"] == "sphere":
            center = torch.tensor(conductor["center"])
            radius = conductor["radius"]
            
            # 生成球面上的点，随机取点并遵循均匀分布 ##############################################
            theta = torch.acos(2 * torch.rand(n_boundary_per_conductor) - 1)
            phi = 2 * np.pi * torch.rand(n_boundary_per_conductor)
            
            x = center[0] + radius * torch.sin(theta) * torch.cos(phi)
            y = center[1] + radius * torch.sin(theta) * torch.sin(phi)
            z = center[2] + radius * torch.cos(theta)
            
            sphere_boundary = torch.stack([x, y, z], dim=1)

            # 存储边界点和导体接地信息
            conductors_boundary_list.append((sphere_boundary, conductor["is_grounded"]))
            
            # 生成球体附近的点 ##############################################
            
            # 随机的扩展系数 0.01-0.05
            near_factors = torch.FloatTensor(n_boundary_per_conductor).uniform_(0.01, 0.05)
            near_radius = radius * (1 + near_factors)
            
            x_near = center[0] + near_radius * torch.sin(theta) * torch.cos(phi)
            y_near = center[1] + near_radius * torch.sin(theta) * torch.sin(phi)
            z_near = center[2] + near_radius * torch.cos(theta)
            
            sphere_near_boundary = torch.stack([x_near, y_near, z_near], dim=1)
            conductors_near_boundary_r.append(sphere_near_boundary)
            
        elif conductor["shape"] == "cylinder":
            center = torch.tensor(conductor["center"])
            radius = conductor["radius"]
            height = conductor["height"]
            half_height = height / 2
            
            # 生成圆柱体表面的点 ##############################################
            n_side = int(0.7 * n_boundary_per_conductor)  # 侧面
            n_caps = n_boundary_per_conductor - n_side    # 顶面和底面
            
            # 侧面点
            theta = 2 * np.pi * torch.rand(n_side)
            h = torch.FloatTensor(n_side).uniform_(-half_height, half_height)
            
            x_side = center[0] + radius * torch.cos(theta)
            y_side = center[1] + radius * torch.sin(theta)
            z_side = center[2] + h
            
            side_points = torch.stack([x_side, y_side, z_side], dim=1)
            
            # 顶面和底面点
            n_per_cap = n_caps // 2
            
            # 生成顶面的点
            r_cap = torch.sqrt(torch.rand(n_per_cap)) * radius
            theta_cap = 2 * np.pi * torch.rand(n_per_cap)
            
            x_top = center[0] + r_cap * torch.cos(theta_cap)
            y_top = center[1] + r_cap * torch.sin(theta_cap)
            z_top = torch.full_like(x_top, center[2] + half_height)
            
            top_points = torch.stack([x_top, y_top, z_top], dim=1)
            
            # 生成底面的点
            x_bottom = center[0] + r_cap * torch.cos(theta_cap)
            y_bottom = center[1] + r_cap * torch.sin(theta_cap)
            z_bottom = torch.full_like(x_bottom, center[2] - half_height)
            
            bottom_points = torch.stack([x_bottom, y_bottom, z_bottom], dim=1)
            
            # 合并所有点
            cylinder_boundary = torch.cat([side_points, top_points, bottom_points], dim=0)
            
            # 存储边界点和导体接地信息
            conductors_boundary_list.append((cylinder_boundary, conductor["is_grounded"]))
            
            # 生成圆柱体附近的点 ##############################################

            # 侧面附近点
            # 扩展半径 radius * 0.01-0.05
            side_near_factors = torch.FloatTensor(n_side).uniform_(0.01, 0.05)
            near_radius = radius * (1 + side_near_factors)
            
            x_near_side = center[0] + near_radius * torch.cos(theta)
            y_near_side = center[1] + near_radius * torch.sin(theta)
            z_near_side = center[2] + h
            
            near_side_points = torch.stack([x_near_side, y_near_side, z_near_side], dim=1)
            
            # 顶面和底面附近点
            # 扩展高 height * 0.01-0.05
            top_near_factors = torch.FloatTensor(n_per_cap).uniform_(0.01, 0.05)
            z_near_top = torch.full_like(x_top, center[2] + half_height) + height * top_near_factors
            near_top_points = torch.stack([x_top, y_top, z_near_top], dim=1)
            
            bottom_near_factors = torch.FloatTensor(n_per_cap).uniform_(0.01, 0.05)
            z_near_bottom = torch.full_like(x_bottom, center[2] - half_height) - height * bottom_near_factors
            near_bottom_points = torch.stack([x_bottom, y_bottom, z_near_bottom], dim=1)
            
            cylinder_near_boundary = torch.cat([near_side_points, near_top_points, near_bottom_points], dim=0)
            conductors_near_boundary_r.append(cylinder_near_boundary)
            
        elif conductor["shape"] == "cube":
            center = torch.tensor(conductor["center"])
            a, b, c = conductor["dimensions"]
            half_a, half_b, half_c = a/2, b/2, c/2

            # 生成立方体表面的点 ##############################################
            
            # 每个面上生成点的数量
            points_per_face_cube = n_boundary_per_conductor // 6
            
            # 生成立方体6个面上的点
            faces = []
            
            # x方向两个面
            y_face = torch.FloatTensor(points_per_face_cube).uniform_(-half_b, half_b)
            z_face = torch.FloatTensor(points_per_face_cube).uniform_(-half_c, half_c)
            x_pos = torch.full_like(y_face, half_a)
            x_neg = torch.full_like(y_face, -half_a)
            
            face_x_pos = torch.stack([x_pos, y_face, z_face], dim=1) + center
            face_x_neg = torch.stack([x_neg, y_face, z_face], dim=1) + center
            faces.extend([face_x_pos, face_x_neg])
            
            # y方向两个面
            x_face = torch.FloatTensor(points_per_face_cube).uniform_(-half_a, half_a)
            z_face_y = torch.FloatTensor(points_per_face_cube).uniform_(-half_c, half_c)
            y_pos = torch.full_like(x_face, half_b)
            y_neg = torch.full_like(x_face, -half_b)
            
            face_y_pos = torch.stack([x_face, y_pos, z_face_y], dim=1) + center
            face_y_neg = torch.stack([x_face, y_neg, z_face_y], dim=1) + center
            faces.extend([face_y_pos, face_y_neg])
            
            # z方向两个面
            x_face_z = torch.FloatTensor(points_per_face_cube).uniform_(-half_a, half_a)
            y_face_z = torch.FloatTensor(points_per_face_cube).uniform_(-half_b, half_b)
            z_pos = torch.full_like(x_face_z, half_c)
            z_neg = torch.full_like(x_face_z, -half_c)
            
            face_z_pos = torch.stack([x_face_z, y_face_z, z_pos], dim=1) + center
            face_z_neg = torch.stack([x_face_z, y_face_z, z_neg], dim=1) + center
            faces.extend([face_z_pos, face_z_neg])
            
            cube_boundary = torch.cat(faces, dim=0)

            # 存储边界点和导体接地信息
            conductors_boundary_list.append((cube_boundary, conductor["is_grounded"]))
            
            # 生成立方体附近的点 ##############################################
            near_faces = []
            
            
            # x方向近边界点
            # 随机偏移
            x_pos_factors = torch.FloatTensor(points_per_face_cube).uniform_(0.01, 0.05)
            x_neg_factors = torch.FloatTensor(points_per_face_cube).uniform_(0.01, 0.05)
            x_pos_near = torch.full_like(y_face, half_a) + a * x_pos_factors
            x_neg_near = torch.full_like(y_face, -half_a) - a * x_neg_factors
            
            face_x_pos_near = torch.stack([x_pos_near, y_face, z_face], dim=1) + center
            face_x_neg_near = torch.stack([x_neg_near, y_face, z_face], dim=1) + center
            near_faces.extend([face_x_pos_near, face_x_neg_near])
            
            # y方向近边界点
            # 随机偏移
            y_pos_factors = torch.FloatTensor(points_per_face_cube).uniform_(0.01, 0.05)
            y_neg_factors = torch.FloatTensor(points_per_face_cube).uniform_(0.01, 0.05)
            y_pos_near = torch.full_like(x_face, half_b) + b * y_pos_factors
            y_neg_near = torch.full_like(x_face, -half_b) - b * y_neg_factors
            
            face_y_pos_near = torch.stack([x_face, y_pos_near, z_face_y], dim=1) + center
            face_y_neg_near = torch.stack([x_face, y_neg_near, z_face_y], dim=1) + center
            near_faces.extend([face_y_pos_near, face_y_neg_near])
            
            # z方向近边界点
            # 随机偏移
            z_pos_factors = torch.FloatTensor(points_per_face_cube).uniform_(0.01, 0.05)
            z_neg_factors = torch.FloatTensor(points_per_face_cube).uniform_(0.01, 0.05)
            z_pos_near = torch.full_like(x_face_z, half_c) + c * z_pos_factors
            z_neg_near = torch.full_like(x_face_z, -half_c) - c * z_neg_factors
            
            face_z_pos_near = torch.stack([x_face_z, y_face_z, z_pos_near], dim=1) + center
            face_z_neg_near = torch.stack([x_face_z, y_face_z, z_neg_near], dim=1) + center
            near_faces.extend([face_z_pos_near, face_z_neg_near])
            
            cube_near_boundary = torch.cat(near_faces, dim=0)
            conductors_near_boundary_r.append(cube_near_boundary)
    
    # 将所有导体近边界点合并
    if conductors_near_boundary_r:
        conductors_near_boundary_r = torch.cat(conductors_near_boundary_r, dim=0)
    else:
        conductors_near_boundary_r = torch.zeros((0, 3))

    # 4. 在每个带电体内部生成采样点 ##################################
    charged_bodies_list = []
    for charged_body in field.charged_bodies:
        if charged_body["shape"] == "sphere":
            volume = (4/3) * np.pi * charged_body["radius"] ** 3
            # 根据体积计算采样点数量
            n_charged_body = max(int(n_density_charged_body * volume), 1000)  # 至少1000个点 
            center = torch.tensor(charged_body["center"])
            radius = charged_body["radius"]
            
            # 生成球体内部的点
            r = radius * torch.rand(n_charged_body) 
            theta = torch.acos(2 * torch.rand(n_charged_body) - 1)
            phi = 2 * np.pi * torch.rand(n_charged_body)
            
            x = center[0] + r * torch.sin(theta) * torch.cos(phi)
            y = center[1] + r * torch.sin(theta) * torch.sin(phi)
            z = center[2] + r * torch.cos(theta)
            
            sphere_inside_points = torch.stack([x, y, z], dim=1)
            
            # 存储带电体的采样点和电荷密度
            charged_bodies_list.append((sphere_inside_points, charged_body["density"]))
    
    return domain_r, far_boundary_r, conductors_boundary_list, conductors_near_boundary_r, charged_bodies_list


def train_pinn(
        model: ElectricPotentialNN,
        field: Field,
        n_rounds: int = 10,
        iters_per_round: int = 1000,
        lr: float = 0.001, 
        comp_domain: tuple = (-10, 10, -10, 10, -10, 10),
        w_pde: float = 1.0,
        w_pde_charged: float = 10.0,
        w_far: float = 10.0,
        w_conductor: float = 10.0
        ) -> list:
    """
    PINN模型的训练函数
    
    Args:
        model: PINN模型
        field: 电场与导体设置
        n_rounds: 训练轮数
        iters_per_round: 每轮训练的迭代次数
        lr: 学习率
        comp_domain: 计算域范围 (xmin, xmax, ymin, ymax, zmin, zmax)
        w_pde: PDE损失权重(无带电体处)
        w_pde_charged: PDE损失权重(带电体处)
        w_far: 极远处边界损失权重
        w_conductor: 导体边界损失权重
    
    Returns:
        list: losses 损失函数历史
    """
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    
    # 初始学习率
    last_lr = lr
    print(f"Initial learning rate: {last_lr}")
    
    # 对孤立导体 #######################
    # 创建可训练电势参数
    isolated_potentials = []
    for conductor in field.conductors:
        if not conductor["is_grounded"]:
            conductor_potential = torch.nn.Parameter(torch.zeros(1, device=device))
            isolated_potentials.append(conductor_potential)
    
    # 添加电势参数到优化器
    if isolated_potentials:
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': isolated_potentials}
        ], lr=lr)

    # 损失历史
    losses = []
    pde_losses = []
    far_boundary_losses = []
    conductor_boundary_losses = []
    charged_body_losses = []
    
    # 轮次训练循环
    total_iters = 0
    for round_idx in range(n_rounds):
        print(f"Starting round {round_idx+1}/{n_rounds}")
        
        # 每轮开始时重新生成采样点
        domain_r, far_boundary_r, conductors_boundary_list, conductors_near_boundary_r, charged_bodies_list = generate_collocation_points(
            field, comp_domain=comp_domain
        )
        
        # 移动到设备
        domain_r = domain_r.to(device)
        conductors_near_boundary_r = conductors_near_boundary_r.to(device)
        
        for i in range(len(conductors_boundary_list)):
            points, is_grounded = conductors_boundary_list[i]
            conductors_boundary_list[i] = (points.to(device), is_grounded)

        for i in range(len(charged_bodies_list)):
            points, density = charged_bodies_list[i]
            charged_bodies_list[i] = (points.to(device), density)

        # 若存在背景电场，则生成远处边界上的点
        if far_boundary_r is not None:
            far_boundary_r = far_boundary_r.to(device)
            # 无穷远处边界点  ##################################
            # 根据field设置的背景电场计算电势值
            far_boundary_potential = torch.zeros(far_boundary_r.shape[0], 1, device=device)
            for i in range(far_boundary_r.shape[0]):
                point = far_boundary_r[i].cpu().numpy()
                phi = torch.tensor(field.to_potential(point), dtype=torch.float32, device=device)
                far_boundary_potential[i, 0] = phi
        
        # 每轮的迭代训练
        for iter_idx in range(iters_per_round):
            optimizer.zero_grad()
            
            # PDE损失

            pde_loss = [torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)] # [无带电体, 带电体]
            
            # 对无带电体处 ∇²φ = 0
            laplacian = model.laplacian(domain_r)
            pde_loss[0] = torch.mean(laplacian ** 2)
            
            # 对带电体处 ∇²φ = -ρ/ε₀
            for points, density in charged_bodies_list:
                if points.shape[0] > 0:  # 有采样点时
                    # 计算电势的拉普拉斯
                    laplacian = model.laplacian(points)
                    
                    charge_density = density / epsilon_0
                    
                    pde_loss[1] += torch.mean((laplacian + charge_density) ** 2)
            
            # 若存在背景电场，则计算远处边界损失
            if far_boundary_r is not None:
                far_pred = model(far_boundary_r)
                far_boundary_loss = torch.mean((far_pred - far_boundary_potential) ** 2)
            else:
                far_boundary_loss = torch.tensor(0.0, device=device)
            
            # 导体边界条件损失
            conductor_boundary_loss = torch.tensor(0.0, device=device)
            isolated_idx = 0
            
            for i, (conductor_points, is_grounded) in enumerate(conductors_boundary_list):
                if conductor_points.shape[0] > 0:  # 有采样点时
                    # 导体表面点的电势
                    conductor_surf_phi_pred = model(conductor_points)
                    
                    if is_grounded:
                        # 接地导体：电势为0
                        target_phi = torch.zeros_like(conductor_surf_phi_pred)
                        conductor_boundary_loss += torch.mean((conductor_surf_phi_pred - target_phi) ** 2)
                    else:
                        # 孤立导体：等势
                        potential = isolated_potentials[isolated_idx]
                        target_phi = potential.expand_as(conductor_surf_phi_pred)
                        conductor_boundary_loss += torch.mean((conductor_surf_phi_pred - target_phi) ** 2)
                        isolated_idx += 1
            
            # 总损失
            loss = (w_pde * pde_loss[0] + w_far * far_boundary_loss +
                w_conductor * conductor_boundary_loss + w_pde_charged * pde_loss[1])
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 记录损失
            losses.append(loss.item())
            pde_losses.append((pde_loss[0].item(), pde_loss[1].item()))
            far_boundary_losses.append(far_boundary_loss.item())
            conductor_boundary_losses.append(conductor_boundary_loss.item())
            
            # 全局迭代计数
            total_iters += 1
            
            # 训练进度
            if (iter_idx + 1) % (max(1, iters_per_round // 5)) == 0 or iter_idx == 0:
                print(f"Round {round_idx+1}/{n_rounds}, Iter {iter_idx+1}/{iters_per_round}, "
                      f"Loss: {loss.item():.6f} "
                      f"PDE: Vacuum: {pde_loss[0].item():.6f}, Charged: {pde_loss[1].item():.6f} "
                      f"Far: {far_boundary_loss.item():.6f}, Conductor: {conductor_boundary_loss.item():.6f}")
        
        # 每轮结束后更新学习率
        scheduler.step(loss)
        
        # 检查并打印学习率变化
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f"Learning rate changed from {last_lr} to {current_lr}")
            last_lr = current_lr
    
    # 记录field中孤立导体的电势值
    if isolated_potentials:
        isolated_idx = 0
        for conductor in field.conductors:
            if not conductor["is_grounded"]:
                conductor["potential"] = isolated_potentials[isolated_idx].item()
                print(f"Isolated conductor{isolated_idx} potential: {conductor['potential']}")
                isolated_idx += 1

    return losses

def visualize_results(
        model: ElectricPotentialNN,
        field: Field,
        comp_domain: tuple = (-10, 10, -10, 10, -10, 10),
        resolution: int = 100,
        z_val: float = 0.0,
        y_val: float = 0.0,
        x_val: float = 0.0
        ):
    """
    可视化PINN模型结果
    
    Args:
        model: 训练好的PINN模型
        field: 电场与导体设置
        comp_domain: 计算域范围 (xmin, xmax, ymin, ymax, zmin, zmax)
        resolution: 网格分辨率
        z_val: xy截面的z值
        y_val: xz截面的y值
        x_val: yz截面的x值
    """
    model.eval()

    xmin, xmax, ymin, ymax, zmin, zmax = comp_domain
    
    fig = plt.figure(figsize=(16, 14))
    
    # 1. 空间电场分布  ##################################
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 使用field_resolution分辨率的网格来显示电场向量
    field_resolution = max(10, resolution // 10)
    x = np.linspace(xmin, xmax, field_resolution)
    y = np.linspace(ymin, ymax, field_resolution)
    z = np.linspace(zmin, zmax, field_resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 提取网格点
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # 过滤导体内部的点
    valid_points = []
    for point in points:
        if not field.is_inside_conductor(point):
            valid_points.append(point)
    
    if valid_points:
        points = np.array(valid_points)
        
        # 将点转换为张量并移至设备
        points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
        
        # 计算电场
        points_tensor.requires_grad_(True)
        E_field = model.compute_electric_field(points_tensor).detach().cpu().numpy()

        # 绘制电场向量图
        ax1.quiver(points[:, 0], points[:, 1], points[:, 2], 
               E_field[:, 0], E_field[:, 1], E_field[:, 2],
               length=0.5, normalize=False, color='b', alpha=0.8, linewidths=0.5)
    
    # 添加导体
    for conductor in field.conductors:
        if conductor["shape"] == "cylinder":
            field.visualize_cylinder(ax1, conductor)
        elif conductor["shape"] == "sphere":
            field.visualize_sphere(ax1, conductor)
        elif conductor["shape"] == "cube":
            field.visualize_cube(ax1, conductor)
    # 添加带电体
    for charged_body in field.charged_bodies:
        if charged_body["shape"] == "sphere":
            field.visualize_sphere(ax1, charged_body, is_charged_body=True)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Electric Field Distribution')
    
    # 2. xy平面电势分布 ##################################
    ax2 = fig.add_subplot(222)
    
    # xy平面网格点
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 输入张量
    xy_points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_val)])
    
    # 计算电势值
    potential_xy = np.zeros(xy_points.shape[0])
    
    # 分批处理
    batch_size = 10000
    for i in range(0, xy_points.shape[0], batch_size):
        batch_points = xy_points[i:i+batch_size]
        
        # 过滤导体内部
        valid_indices = [j for j, point in enumerate(batch_points) if not field.is_inside_conductor(point)]
        
        if valid_indices:
            valid_batch = batch_points[valid_indices]
            valid_batch_tensor = torch.tensor(valid_batch, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pot_values = model(valid_batch_tensor).cpu().numpy().flatten()
            
            for j, idx in enumerate(valid_indices):
                potential_xy[i + idx] = pot_values[j]
    
    # 网格形状
    potential_xy = potential_xy.reshape(X.shape)
    
    # 电势图
    contour = ax2.contourf(X, Y, potential_xy, 50, cmap='viridis')
    fig.colorbar(contour, ax=ax2, label='Electric Potential (V)')
    
    # 标记导体的xy截面
    for conductor in field.conductors:
        if conductor["shape"] == "cylinder":
            cx, cy, cz = conductor["center"]
            r = conductor["radius"]
            h = conductor["height"]
            half_h = h / 2
            
            if cz - half_h <= z_val <= cz + half_h:
                circle = plt.Circle((cx, cy), r, fill=False, color='blue')
                ax2.add_patch(circle)
        
        elif conductor["shape"] == "sphere":
            cx, cy, cz = conductor["center"]
            r = conductor["radius"]
            
            # 计算在z=z_val平面上的圆的半径
            z_diff = abs(z_val - cz)
            if z_diff <= r:
                circle_r = (r**2 - z_diff**2)**0.5
                circle = plt.Circle((cx, cy), circle_r, fill=False, color='blue')
                ax2.add_patch(circle)
        
        elif conductor["shape"] == "cube":
            cx, cy, cz = conductor["center"]
            a, b, c = conductor["dimensions"]
            half_a, half_b, half_c = a/2, b/2, c/2
            
            if cz - half_c <= z_val <= cz + half_c:
                rect = plt.Rectangle((cx - half_a, cy - half_b), a, b, fill=False, color='blue')
                ax2.add_patch(rect)
    
    # 标记带电体的xy截面
    for charged_body in field.charged_bodies:
        if charged_body["shape"] == "sphere":
            cx, cy, cz = charged_body["center"]
            r = charged_body["radius"]
            
            # 计算在z=z_val平面上的圆的半径
            z_diff = abs(z_val - cz)
            if z_diff <= r:
                circle_r = (r**2 - z_diff**2)**0.5
                circle = plt.Circle((cx, cy), circle_r, fill=False, color='red')
                ax2.add_patch(circle)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'XY Plane Electric Potential Distribution (Z={z_val})')
    ax2.set_aspect('equal')
    
    # 3. xz平面电势分布 ##################################
    ax3 = fig.add_subplot(223)
    
    # xz平面网格点
    x = np.linspace(xmin, xmax, resolution)
    z = np.linspace(zmin, zmax, resolution)
    X, Z = np.meshgrid(x, z)
    
    # 输入张量
    xz_points = np.column_stack([X.ravel(), np.full(X.size, y_val), Z.ravel()])
    
    # 电势值
    potential_xz = np.zeros(xz_points.shape[0])
    
    # 分批处理
    for i in range(0, xz_points.shape[0], batch_size):
        batch_points = xz_points[i:i+batch_size]
        valid_indices = [j for j, point in enumerate(batch_points) if not field.is_inside_conductor(point)]
        
        if valid_indices:
            valid_batch = batch_points[valid_indices]
            valid_batch_tensor = torch.tensor(valid_batch, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pot_values = model(valid_batch_tensor).cpu().numpy().flatten()
            
            for j, idx in enumerate(valid_indices):
                potential_xz[i + idx] = pot_values[j]
    
    # 网格形状
    potential_xz = potential_xz.reshape(X.shape)
    
    # 电势图
    contour = ax3.contourf(X, Z, potential_xz, 50, cmap='viridis')
    fig.colorbar(contour, ax=ax3, label='Electric Potential (V)')
    
    # 标记导体的xz截面
    for conductor in field.conductors:
        if conductor["shape"] == "cylinder":
            cx, cy, cz = conductor["center"]
            r = conductor["radius"]
            h = conductor["height"]
            half_h = h / 2
            
            # 若圆柱体与y=y_val平面相交
            y_diff = abs(y_val - cy)
            if y_diff <= r:
                # 矩形截面
                rect_width = 2 * (r**2 - y_diff**2)**0.5
                rect = plt.Rectangle((cx - rect_width/2, cz - half_h), rect_width, h, fill=False, color='blue')
                ax3.add_patch(rect)
        
        elif conductor["shape"] == "sphere":
            cx, cy, cz = conductor["center"]
            r = conductor["radius"]
            
            # 在y=y_val平面上的圆半径
            y_diff = abs(y_val - cy)
            if y_diff <= r:
                circle_r = (r**2 - y_diff**2)**0.5
                circle = plt.Circle((cx, cz), circle_r, fill=False, color='blue')
                ax3.add_patch(circle)
        
        elif conductor["shape"] == "cube":
            cx, cy, cz = conductor["center"]
            a, b, c = conductor["dimensions"]
            half_a, half_b, half_c = a/2, b/2, c/2
            
            if cy - half_b <= y_val <= cy + half_b:
                rect = plt.Rectangle((cx - half_a, cz - half_c), a, c, fill=False, color='blue')
                ax3.add_patch(rect)

    # 标记带电体的xz截面
    for charged_body in field.charged_bodies:
        if charged_body["shape"] == "sphere":
            cx, cy, cz = charged_body["center"]
            r = charged_body["radius"]
            
            # 在y=y_val平面上的圆半径
            y_diff = abs(y_val - cy)
            if y_diff <= r:
                circle_r = (r**2 - y_diff**2)**0.5
                circle = plt.Circle((cx, cz), circle_r, fill=False, color='red')
                ax3.add_patch(circle)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title(f'XZ Plane Electric Potential Distribution (Y={y_val})')
    ax3.set_aspect('equal')
    
    # 4. yz平面电势分布 ##################################
    ax4 = fig.add_subplot(224)
    
    # yz平面网格点
    y = np.linspace(ymin, ymax, resolution)
    z = np.linspace(zmin, zmax, resolution)
    Y, Z = np.meshgrid(y, z)
    
    # 输入张量
    yz_points = np.column_stack([np.full(Y.size, x_val), Y.ravel(), Z.ravel()])
    
    # 电势值
    potential_yz = np.zeros(yz_points.shape[0])
    
    # 分批处理
    for i in range(0, yz_points.shape[0], batch_size):
        batch_points = yz_points[i:i+batch_size]
        valid_indices = [j for j, point in enumerate(batch_points) if not field.is_inside_conductor(point)]
        
        if valid_indices:
            valid_batch = batch_points[valid_indices]
            valid_batch_tensor = torch.tensor(valid_batch, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                pot_values = model(valid_batch_tensor).cpu().numpy().flatten()
            
            for j, idx in enumerate(valid_indices):
                potential_yz[i + idx] = pot_values[j]
    
    # 网格形状
    potential_yz = potential_yz.reshape(Y.shape)
    
    # 电势图
    contour = ax4.contourf(Y, Z, potential_yz, 50, cmap='viridis')
    fig.colorbar(contour, ax=ax4, label='Electric Potential (V)')
    
    # 标记导体的yz截面
    for conductor in field.conductors:
        if conductor["shape"] == "cylinder":
            cx, cy, cz = conductor["center"]
            r = conductor["radius"]
            h = conductor["height"]
            half_h = h / 2
            
            # 若圆柱体与x=x_val平面相交
            x_diff = abs(x_val - cx)
            if x_diff <= r:
                # 矩形截面
                rect_width = 2 * (r**2 - x_diff**2)**0.5
                rect = plt.Rectangle((cy - rect_width/2, cz - half_h), rect_width, h, fill=False, color='blue')
                ax4.add_patch(rect)
        
        elif conductor["shape"] == "sphere":
            cx, cy, cz = conductor["center"]
            r = conductor["radius"]
            
            # 在x=x_val平面上的圆半径
            x_diff = abs(x_val - cx)
            if x_diff <= r:
                circle_r = (r**2 - x_diff**2)**0.5
                circle = plt.Circle((cy, cz), circle_r, fill=False, color='blue')
                ax4.add_patch(circle)
        
        elif conductor["shape"] == "cube":
            cx, cy, cz = conductor["center"]
            a, b, c = conductor["dimensions"]
            half_a, half_b, half_c = a/2, b/2, c/2
            
            if cx - half_a <= x_val <= cx + half_a:
                rect = plt.Rectangle((cy - half_b, cz - half_c), b, c, fill=False, color='blue')
                ax4.add_patch(rect)

    # 标记带电体的yz截面
    for charged_body in field.charged_bodies:
        if charged_body["shape"] == "sphere":
            cx, cy, cz = charged_body["center"]
            r = charged_body["radius"]
            
            # 在x=x_val平面上的圆半径
            x_diff = abs(x_val - cx)
            if x_diff <= r:
                circle_r = (r**2 - x_diff**2)**0.5
                circle = plt.Circle((cy, cz), circle_r, fill=False, color='red')
                ax4.add_patch(circle)
    
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title(f'YZ Plane Electric Potential Distribution (X={x_val})')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('electric_field_results.png')
    

def main():

    # 设置电场和导体
    field = set_field()
    field.visualize()

    # 区域范围
    xmin, xmax, ymin, ymax, zmin, zmax = field.determine_comp_domain()
    print(f'Computation domain: ({xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax})')
    
    # 网络参数
    hidden_layers = [128, 128, 128, 128, 128, 128]
    learning_rate = 1e-3
    n_rounds = 10
    iters_per_round = 100
    
    # 不同部分损失权重
    w_pde = 1.0
    w_pde_charged = 10.0
    w_far = 10.0
    w_conductor = 10.0

    # 创建PINN模型
    model = ElectricPotentialNN(input_dim=3, hidden_layers=hidden_layers).to(device)

    # 训练PINN模型
    losses = train_pinn(model, field, n_rounds=n_rounds, iters_per_round=iters_per_round, lr=learning_rate,
                        comp_domain=(xmin, xmax, ymin, ymax, zmin, zmax),
                        w_pde=w_pde, w_pde_charged=w_pde_charged,
                        w_far=w_far, w_conductor=w_conductor)
    
    # 训练损失图
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.title('Training Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    # plt.savefig('training_loss_history.png')

    # 结果可视化
    visualize_results(model, field, comp_domain=(xmin, xmax, ymin, ymax, zmin, zmax))


if __name__ == "__main__":
    main()
import torch
import numpy as np
import time
from abc import abstractmethod
from skimage.io import imsave
from tqdm import trange
from argparse import ArgumentParser


def read_obj(path):
    verts = []
    faces = []
    for line in open(path).readlines():
        strips = line.strip().split(' ')
        if strips[0] == 'v':
            verts.append([float(s) for s in strips[1:4]])
        elif strips[0] == 'f':
            faces.append([int(s.split('/')[0]) for s in strips[1:4]])
    return np.array(verts).astype(np.float32), np.array(faces).astype(np.int32)


def generate_rotmat(x=0, y=0, z=0):
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    R = Rx.dot(Ry.dot(Rz))
    return R


def rotate_mesh(verts, x=0, y=0, z=0):
    R = generate_rotmat(x, y, z)
    return verts.dot(R.T)


def rand_cos_vector(size, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    randphi = torch.rand(size, device=device) * 2 * np.pi
    sqcostheta = torch.rand(size, device=device)
    costheta = torch.sqrt(sqcostheta)
    sintheta = torch.sqrt(1 - costheta ** 2)
    res = torch.stack([costheta, sintheta * torch.cos(randphi), sintheta * torch.sin(randphi)], dim=1)
    return res


def rand_cos_vector_by_basis(basis):
    res = rand_cos_vector(basis.size(0))
    return (basis * res.unsqueeze(-1)).sum(dim=1)


class Camera:
    def __init__(self, size, focal_dist, rotation=None, translation=None, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.size = size
        self.focal_dist = (focal_dist, focal_dist)
        self.center = [s / 2 for s in size]
        self.rotmat = rotation
        self.tvec = translation
        if rotation is None:
            self.rotmat = torch.eye(3, device=device)
        if translation is None:
            self.tvec = torch.zeros(3, device=device)

    def cast_rays(self, add_rand=True):
        total_size = self.size[0] * self.size[1]
        grid_x = torch.arange(self.size[0], dtype=torch.float, device=self.device) - self.center[0]
        grid_y = torch.arange(self.size[1], dtype=torch.float, device=self.device) - self.center[1]
        if add_rand:
            grid_x += torch.randn(self.size[0], device=self.device)
            grid_y += torch.randn(self.size[1], device=self.device)
        dirs_x = grid_x / self.focal_dist[0]
        dirs_y = grid_y / self.focal_dist[1]
        dirs_y, dirs_x = torch.meshgrid((-dirs_y, dirs_x))
        dirs_z = torch.ones(total_size, device=self.device)
        dirs_x = dirs_x.contiguous()
        dirs_y = dirs_y.contiguous()
        dirs = torch.stack([dirs_x.view(-1), dirs_y.view(-1), dirs_z], dim=-1)
        dirs = dirs / torch.norm(dirs, p=2, dim=-1, keepdim=True)
        dirs = torch.matmul(dirs, self.rotmat.t())
        coords = self.tvec.clone().view(1, 3).repeat(total_size, 1)
        return coords, dirs


class Structure:
    def __init__(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    @abstractmethod
    def hit(self, ray_coords, ray_dirs):
        pass


class Material:
    def __init__(self):
        pass

    @abstractmethod
    def islight(self):
        pass

    @abstractmethod
    def getluminance(self):
        pass


class SolidMaterial(Material):
    def __init__(self, color=(1., 0., 0.)):
        super().__init__()
        self.color = [float(c) for c in color]

    def islight(self):
        return False

    def getluminance(self):
        return self.color


class Light(Material):
    def __init__(self, flux=30.):
        super().__init__()
        self.flux = flux

    def islight(self):
        return True

    def getluminance(self):
        return [self.flux for _ in range(3)]


class SceneObject:
    def __init__(self, structure: Structure, material: Material):
        self.structure = structure
        self.material = material

    def hit(self, *args, **kwargs):
        return self.structure.hit(*args, **kwargs)


class Mesh(Structure):
    EPS = 1e-8

    def __init__(self, verts, faces, device=None):
        super().__init__(device)
        self.verts = torch.tensor(verts, device=self.device).float()
        self.faces = torch.tensor(faces, device=self.device).long() - 1
        self.plane_verts = self.verts[self.faces.view(-1), :].view(self.faces.size(0), 3, 3)  # Nx3x3
        self.plane_edges = torch.stack([self.plane_verts[:, 1] - self.plane_verts[:, 0],
                                        self.plane_verts[:, 2] - self.plane_verts[:, 1],
                                        self.plane_verts[:, 0] - self.plane_verts[:, 2]], dim=1)  # Nx3x3
        # print(self.verts.size(), self.faces.size(), self.plane_verts.size(), self.plane_edges.size())
        self.plane_norms = torch.cross(self.plane_edges[:, 0], -self.plane_edges[:, 2], dim=1)  # Nx3
        plane_v = self.plane_edges[:, 0]
        plane_u = torch.cross(self.plane_norms, plane_v, dim=1)
        self.plane_basis = torch.stack([self.plane_norms, plane_v, plane_u], dim=-2)  # Nx3x3
        # print(self.plane_basis.size())
        # print(torch.norm(self.plane_basis, p=2, dim = -1).size())
        self.plane_basis /= torch.norm(self.plane_basis, p=2, dim=-1).unsqueeze(-1)
        self.plane_dist = -torch.matmul(self.plane_norms.view(-1, 1, 3), self.plane_verts[:, 0].view(-1, 3, 1)).view(
            -1)  # N
        # print(self.plane_dist)

    def hit(self, ray_coords, ray_dirs):
        norm_dot_raydir = torch.matmul(ray_dirs, self.plane_norms.t())  # MxN
        non_parallel_mask = torch.abs(norm_dot_raydir) > Mesh.EPS  # MxN
        intersect_point = -(torch.matmul(ray_coords, self.plane_norms.t()) + self.plane_dist) / norm_dot_raydir  # MxN
        non_behind_mask = intersect_point > 0  # MxN
        global_intersect_point = ray_coords.unsqueeze(1) + ray_dirs.unsqueeze(1) * intersect_point.unsqueeze(
            -1)  # MxNx3

        int_to_vs = global_intersect_point.unsqueeze(2) - self.plane_verts.unsqueeze(0)  # MxNx3x3
        cross_prods = torch.cross(self.plane_edges.unsqueeze(0).repeat(int_to_vs.size(0), 1, 1, 1), int_to_vs,
                                  dim=3)  # MxNx3x3
        triangle_side = torch.matmul(cross_prods, self.plane_norms.view(1, -1, 3, 1)).squeeze(-1)  # MxNx3
        inside_mask = (triangle_side >= 0).min(dim=2)[0]  # MxN
        res_mask = inside_mask & non_behind_mask & non_parallel_mask
        hit_depths = torch.where(res_mask, intersect_point, torch.ones_like(intersect_point) * float('inf'))  # MxN
        hit_inds = torch.argmin(hit_depths, dim=1)  # M
        hit_mask = res_mask.max(dim=-1)[0]
        hit_depths = torch.gather(hit_depths, 1, hit_inds.view(-1, 1)).squeeze(-1)
        hit_points = torch.gather(global_intersect_point, 1, hit_inds.view(-1, 1, 1).repeat(1, 1, 3)).squeeze(1)
        hit_basis = self.plane_basis[hit_inds, :, :]
        # print("Hit mask:", hit_mask)
        return hit_mask, hit_depths, hit_points, hit_basis


class Scene:
    COORD_MOVING_REL = 1e-3

    def __init__(self, camera: Camera, objects=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if objects is None:
            self.objects = []
        self.camera = camera

    def build_scene(self):
        self.islight = torch.tensor([o.material.islight() for o in self.objects], device=self.device)
        self.luminance = torch.tensor([o.material.getluminance() for o in self.objects], device=self.device)

    def check_intersections(self, ray_coords, ray_dirs):
        hit_mask = torch.zeros(ray_coords.size(0), dtype=torch.bool, device=self.device)
        hit_depths = torch.ones(ray_coords.size(0), device=self.device) * float('inf')
        hit_points = torch.zeros_like(ray_coords)
        hit_basis = torch.zeros(ray_dirs.size(0), 3, 3, device=self.device)
        obj_inds = torch.zeros(ray_coords.size(0), dtype=torch.long, device=self.device)
        for ind, obj in enumerate(self.objects):
            mask, depths, points, basis = obj.hit(ray_coords, ray_dirs)
            hit_mask |= mask
            lessdepth_mask = hit_depths > depths
            hit_depths[lessdepth_mask] = depths[lessdepth_mask]
            hit_points[lessdepth_mask, :] = points[lessdepth_mask, :]
            hit_basis[lessdepth_mask, ...] = basis[lessdepth_mask, ...]
            obj_inds[lessdepth_mask] = ind
        hit_basis[:, 0, :] *= -torch.sign(
            torch.matmul(ray_dirs.unsqueeze(1), hit_basis[:, 0, :].unsqueeze(-1)).squeeze(-1))
        return hit_mask, hit_points, hit_basis, obj_inds

    def render(self, maxdepth=5, add_rand = True):
        ray_coords, ray_dirs = self.camera.cast_rays(add_rand)
        nondone_mask = torch.ones(ray_coords.size(0), dtype=torch.bool, device=self.device)
        reflectance = torch.ones_like(ray_coords)
        color = torch.zeros_like(ray_coords)
        for render_depth in range(maxdepth):
            hit_mask, hit_points, hit_basis, obj_inds = self.check_intersections(ray_coords, ray_dirs)
            # print(hit_basis[:,0,:])
            # print(ray_dirs)

            light_mask = self.islight[obj_inds]
            hit_vals = self.luminance[obj_inds]
            color_mask = nondone_mask.clone()
            reflectance_mask = nondone_mask.clone()
            local_color_mask = hit_mask & light_mask
            color_mask[nondone_mask] = local_color_mask
            local_reflectance_mask = hit_mask & (~light_mask)
            reflectance_mask[nondone_mask] = local_reflectance_mask
            color[color_mask] = reflectance[color_mask] * hit_vals[local_color_mask]

            if local_reflectance_mask.sum() == 0:
                break
            reflectance[reflectance_mask] = reflectance[reflectance_mask] * hit_vals[local_reflectance_mask]
            ray_coords = hit_points[local_reflectance_mask]
            ray_dirs = rand_cos_vector_by_basis(hit_basis[local_reflectance_mask, ...])
            ray_coords += ray_dirs * Scene.COORD_MOVING_REL
            nondone_mask = reflectance_mask

        return color.view(self.camera.size[1], self.camera.size[0], 3)#.transpose(0, 1)

    def add_object(self, structure, material):
        self.objects.append(SceneObject(structure, material))


if __name__ == '__main__':
    parser = ArgumentParser(description="PyTorch RayTracer")
    parser.add_argument("output")
    parser.add_argument("-i", "--iter_num", default=1000, type=int)
    parser.add_argument("-is", "--intermediate_saves", default=50, type=int)
    parser.add_argument("-res", "--resolution", default=(600, 400), nargs=2, type=int)
    parser.add_argument("-f", "--focal_dist", default=3e2, type=float)
    args = parser.parse_args()

    stime = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera = Camera(args.resolution, args.focal_dist, translation=torch.tensor([0, 0, -1.5], device=device), device=device)

    # Scene description
    scene = Scene(camera, device=device)

    cube_v, cube_f = read_obj('models/cube.obj')
    cube_v = (cube_v - 0.5) * 4
    scene.add_object(Mesh(cube_v.copy(), cube_f, device=device), SolidMaterial((0, 0.7, 0)))

    cube_v, cube_f = read_obj('models/cube.obj')
    cube_v = rotate_mesh(cube_v, np.pi / 4, 0, np.pi / 6)
    cube_v -= 0.5
    cube_v /= 2
    cube_v[:, 0] -= 0.5
    scene.add_object(Mesh(cube_v, cube_f, device=device), SolidMaterial((0, 0, 0.9)))

    cube_v, cube_f = read_obj('models/cube.obj')
    cube_v = rotate_mesh(cube_v, -7*np.pi / 8, 0, -np.pi / 3)
    cube_v -= 0.75
    cube_v /= 3
    cube_v[:, 0] += 0.5
    cube_v[:, 2] += 0.7
    scene.add_object(Mesh(cube_v, cube_f, device=device), SolidMaterial((0.8, 0.4, 0)))

    # obj_v, obj_f = read_obj('models/teddy.obj')
    # obj_v = rotate_mesh(obj_v, -np.pi / 4, 0, np.pi / 8)
    # obj_v /= 32
    # # obj_v -= 0.5
    # obj_v[:, 0] += 0.75
    # scene.add_object(Mesh(obj_v, obj_f, device=device), SolidMaterial((0.95, 0, 0.4)))

    cube_v, cube_f = read_obj('models/plane.obj')
    cube_v = rotate_mesh(cube_v, np.pi / 2, 0, 0)
    cube_v = cube_v/2
    cube_v[:, 1] += 1
    cube_v[:, 0] -= 0.25
    scene.add_object(Mesh(cube_v.copy(), cube_f, device=device), Light(60))

    scene.build_scene()
    print("Initialisation complete, took {:.3f}s".format(time.time() - stime))
    rtime = time.time()
    SAVE_COUNT = 10

    img = None
    with torch.no_grad():
        for iter_num in trange(args.iter_num):
            iter_img = scene.render()
            if img is None:
                img = iter_img
            else:
                img = (img * iter_num + iter_img) / (iter_num + 1)
            if args.intermediate_saves > 0 and (iter_num + 1) % args.intermediate_saves == 0:
                imsave(args.output, (img.clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8))
        img = torch.clamp(img, 0., 1.)
    print("Rendering completed, took {:.3f}s".format(time.time() - rtime))
    imsave(args.output, (img.cpu().numpy() * 255).astype(np.uint8))
    print("Overall time {:.3f}s".format(time.time() - stime))

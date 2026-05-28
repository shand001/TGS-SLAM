    
    
import torch   
import numpy as np
def load_bound(config):
    """
    Pass the scene bound parameters to different decoders and 

    Args:
        cfg (dict): parsed config dict.
    """

    scale = config.get("scale",1)
    # scale the bound if there is a global scaling factor
    bound = torch.from_numpy(np.array(config['mapping']['bound'])*scale).float()
    bound_dividable = config['c_planes_res']['bound_dividable']
    # enlarge the bound a bit to allow it dividable by bound_dividable
    bound[:, 1] = (((bound[:, 1]-bound[:, 0]) /
                        bound_dividable).int()+1)*bound_dividable+bound[:, 0]
    return bound


def init_planes(cfg, bound):
    """
    Initialize the feature planes.

    Args:
        cfg (dict): parsed config dict.
    """


    coarse_s_planes_res = cfg['s_planes_res']['coarse']
    fine_s_planes_res = cfg['s_planes_res']['fine']
    

    c_dim = cfg['model']['c_dim']
    xyz_len = bound[:, 1]-bound[:, 0]

    ####### Initializing Planes ############

    s_planes_xy, s_planes_xz, s_planes_yz = [], [], []
    s_planes_res = [coarse_s_planes_res, fine_s_planes_res]
    planes_dim = c_dim

    for grid_res in s_planes_res:
        grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
        grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
        s_planes_xy.append(torch.empty(
            [1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
        s_planes_xz.append(torch.empty(
            [1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
        s_planes_yz.append(torch.empty(
            [1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

    coarse_c_planes_res = cfg['c_planes_res']['coarse']
    fine_c_planes_res = cfg['c_planes_res']['fine']

    c_planes_xy, c_planes_xz, c_planes_yz = [], [], []
    c_planes_res = [coarse_c_planes_res, fine_c_planes_res]
    for grid_res in c_planes_res:
        grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
        grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
        c_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
        c_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
        c_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))
    

    return (c_planes_xy, c_planes_xz, c_planes_yz, s_planes_xy, s_planes_xz, s_planes_yz)

def normalize_3d_coordinate(p, bound):
    """
    Normalize 3d coordinate to [-1, 1] range.
    Args:
        p: (N, 3) 3d coordinate
        bound: (3, 2) min and max of each dimension
    Returns:
        (N, 3) normalized 3d coordinate

    """
    # print("p:", p.device, p.dtype, p.shape)
    # print("bound:", getattr(bound, "device", "not tensor"),
    #   getattr(bound, "dtype", "not tensor"))
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p
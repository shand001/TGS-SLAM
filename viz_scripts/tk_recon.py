import os
import sys
import json
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

from viz_scripts.final_recon import load_camera, load_scene_data, make_lineset, render, rgbd2pcd
from utils.slam_external import build_rotation
from utils.slam_helpers import get_depth_and_silhouette
from utils.recon_helpers import setup_camera
from utils.common_utils import seed_everything
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from enum import Enum
from queue import Queue
from threading import Thread
from tkinter import filedialog, messagebox
import tkinter as tk
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import argparse


# Queue for inter-thread communication
command_queue = Queue()


def extract_semantic_ids_from_params(scene_path):
    """
    Extract available semantic ids from params.npz without moving data to GPU.
    """
    try:
        params = np.load(scene_path, allow_pickle=True)
    except Exception as exc:
        print(f"Could not read semantic ids from {scene_path}: {exc}")
        return set()

    if 'semantic_id' in params:
        sem = params['semantic_id']
        if sem.ndim > 1:
            labels = np.argmax(sem, axis=-1)
        else:
            labels = sem
        return set(np.unique(labels.astype(np.int64)).tolist())

    if 'semantic_ids' in params:
        sem = params['semantic_ids']
        return set(np.unique(sem.astype(np.int64)).tolist())

    return set()


class Operation(Enum):
    SWITCH_MODE = 1
    APPLY_MASK = 2
    CAM_TRANS = 3
    CAM_ROTATE = 4
    SEM_TRANS = 5
    SEM_ROTATE = 6
    SEM_DELETE = 7


class ControlPanel(tk.Frame):
    def __init__(self, master, command_queue, cfg, width, height, semantic_ids=None, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.master = master
        self.command_queue = command_queue
        master.title('Control Panel')
        self.default_mode = cfg.get('render_mode', 'color')

        # Semantic IDs
        self.semantic_ids_label = tk.Label(master, text='Semantic IDs:')
        self.semantic_ids_label.pack()

        self.sence_name = cfg['scene_name']
        self.semantic_ids = set(
            sorted(semantic_ids)) if semantic_ids else set()
        self.current_semantic_id = None

        semantic_ids_display = ', '.join(str(num) for num in sorted(
            self.semantic_ids)) if self.semantic_ids else 'Any'
        self.semantic_ids_value = tk.Label(master, text=semantic_ids_display)
        self.semantic_ids_value.pack()
        self.current_semantic_label = tk.Label(master, text='Current ID: None')
        self.current_semantic_label.pack()

        # Mode
        self.mode_label = tk.Label(master, text='Mode:')
        self.mode_label.pack()

        self.mode_var = tk.StringVar(value=self.default_mode)  # Default value
        self.colors_radiobutton = tk.Radiobutton(
            master, text='Colors', variable=self.mode_var, value='color')
        self.colors_radiobutton.pack(anchor=tk.CENTER)

        self.centers_radiobutton = tk.Radiobutton(
            master, text='Centers', variable=self.mode_var, value='centers')
        self.centers_radiobutton.pack(anchor=tk.CENTER)

        self.semantic_colors_radiobutton = tk.Radiobutton(
            master, text='Semantic Colors', variable=self.mode_var, value='semantic_color')
        self.semantic_colors_radiobutton.pack(anchor=tk.CENTER)

        # Manipulate
        self.manipulate_label = tk.Label(master, text='Manipulate:')
        self.manipulate_label.pack()

        self.semantic_id_entry_label = tk.Label(master, text='Semantic ID:')
        self.semantic_id_entry_label.pack()

        self.semantic_id_entry = tk.Entry(master)
        self.semantic_id_entry.pack()

        self.keep_var = tk.BooleanVar(value=True)
        self.keep_checkbox = tk.Checkbutton(
            master, text='Keep', variable=self.keep_var)
        self.keep_checkbox.pack()

        # Create buttons for moving and rotating
        self.create_buttons("Transition", ["X", "Y", "Z"], "move", "-", "+")
        self.create_buttons("Rotation", ["X", "Y", "Z"], "rotate", "-", "+")
        self.create_semantic_buttons(
            "Object Transition", ["X", "Y", "Z"], "sem_move", "-", "+")
        self.create_semantic_buttons(
            "Object Rotation", ["X", "Y", "Z"], "sem_rotate", "-", "+")

        # Delete current semantic selection
        self.delete_button = tk.Button(
            master, text='Delete Semantic', command=self.delete_semantic)
        self.delete_button.pack()

        # Apply / Reset buttons
        self.apply_button = tk.Button(master, text='Apply', command=self.apply)
        self.apply_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(master, text='Reset', command=self.reset)
        self.reset_button.pack(side=tk.RIGHT)

    def apply(self):
        # Here you would handle the application of the settings
        mode = self.mode_var.get()
        semantic_id_text = self.semantic_id_entry.get().strip()
        keep = self.keep_var.get()
        # messagebox.showinfo('Apply', f'Applied settings:\nMode: {mode}\nSemantic ID: {semantic_id}\nKeep: {keep}')
        cmd = {'type': Operation.SWITCH_MODE, 'payload': {'mode': mode}}
        print("Apply: ", cmd)
        self.command_queue.put(cmd)

        if semantic_id_text == '':
            semantic_id_val = -1  # no-op
        elif semantic_id_text.isdigit():
            semantic_id_val = int(semantic_id_text)
        else:
            semantic_id_val = None

        if semantic_id_val is None:
            print("Invalid Apply: ", semantic_id_text)
            return

        if self.semantic_ids and semantic_id_val not in self.semantic_ids:
            print(
                f"Semantic id {semantic_id_val} not in available set {self.semantic_ids}")
            return

        self.current_semantic_id = semantic_id_val
        self.current_semantic_label.configure(
            text=f"Current ID: {semantic_id_val if semantic_id_val >= 0 else 'None'}")

        cmd = {'type': Operation.APPLY_MASK, 'payload': {
            'semantic_id': semantic_id_val,
            'to_keep': keep
        }
        }
        self.command_queue.put(cmd)

    def create_buttons(self, action, directions, transform_type, minus_text, plus_text):
        action_label = tk.Label(self.master, text=f"{action}:")
        action_label.pack()

        for direction in directions:
            button_frame = tk.Frame(self.master)
            button_frame.pack()

            label = tk.Label(button_frame, text=f"{direction}:")
            label.pack(side=tk.LEFT)

            minus_button = tk.Button(
                button_frame, text=f"{minus_text}", command=lambda d=direction: self.perform_transform(transform_type, d, -1))
            minus_button.pack(side=tk.LEFT)

            plus_button = tk.Button(
                button_frame, text=f"{plus_text}", command=lambda d=direction: self.perform_transform(transform_type, d, 1))
            plus_button.pack(side=tk.LEFT)

    def create_semantic_buttons(self, action, directions, transform_type, minus_text, plus_text):
        action_label = tk.Label(self.master, text=f"{action}:")
        action_label.pack()

        for direction in directions:
            button_frame = tk.Frame(self.master)
            button_frame.pack()

            label = tk.Label(button_frame, text=f"{direction}:")
            label.pack(side=tk.LEFT)

            minus_button = tk.Button(
                button_frame, text=f"{minus_text}", command=lambda d=direction: self.perform_semantic_transform(transform_type, d, -1))
            minus_button.pack(side=tk.LEFT)

            plus_button = tk.Button(
                button_frame, text=f"{plus_text}", command=lambda d=direction: self.perform_semantic_transform(transform_type, d, 1))
            plus_button.pack(side=tk.LEFT)

    def perform_transform(self, transform_type, direction, factor):
        if transform_type == 'move':
            self.command_queue.put({
                'type': Operation.CAM_TRANS,
                'payload': {
                    'direction': direction,
                    'factor': factor,
                }
            })
        elif transform_type == 'rotate':
            self.command_queue.put({
                'type': Operation.CAM_ROTATE,
                'payload': {
                    'direction': direction,
                    'factor': factor,
                }
            })

    def perform_semantic_transform(self, transform_type, direction, factor):
        if self.current_semantic_id is None or self.current_semantic_id < 0:
            print("No semantic id selected for editing.")
            return
        if transform_type == 'sem_move':
            self.command_queue.put({
                'type': Operation.SEM_TRANS,
                'payload': {
                    'direction': direction,
                    'factor': factor,
                    'semantic_id': self.current_semantic_id,
                }
            })
        elif transform_type == 'sem_rotate':
            self.command_queue.put({
                'type': Operation.SEM_ROTATE,
                'payload': {
                    'direction': direction,
                    'factor': factor,
                    'semantic_id': self.current_semantic_id,
                }
            })

    def delete_semantic(self):
        if self.current_semantic_id is None or self.current_semantic_id < 0:
            print("No semantic id selected for deletion.")
            return
        self.command_queue.put({
            'type': Operation.SEM_DELETE,
            'payload': {
                'semantic_id': self.current_semantic_id
            }
        })

    def reset(self):
        # Here you would handle the resetting of the settings to their defaults
        self.mode_var.set(self.default_mode)
        self.semantic_id_entry.delete(0, tk.END)
        self.keep_var.set(False)
        # messagebox.showinfo('Reset', 'Settings have been reset to default values.')
        self.command_queue.put({
            'type': Operation.SWITCH_MODE,
            'payload': {
                'mode': self.default_mode,
            }
        })
        self.command_queue.put({
            'type': Operation.APPLY_MASK,
            'payload': {
                'semantic_id': -2,
            }
        })


def move_camera_x(current_pose, distance):
    """
    Move the camera along the X-axis by a gven distance.
    """
    new_pose = current_pose.copy()
    translation_matrix = np.identity(4)
    translation_matrix[0, 3] = distance
    new_pose = new_pose @ translation_matrix
    return new_pose


def move_camera_y(current_pose, distance):
    """
    Move the camera along the Y-axis by a gven distance.
    """
    new_pose = current_pose.copy()
    translation_matrix = np.identity(4)
    translation_matrix[1, 3] = distance
    new_pose = new_pose @ translation_matrix
    return new_pose


def move_camera_z(current_pose, distance):
    """
    Move the camera along the Z-axis by a gven distance.
    """
    new_pose = current_pose.copy()
    translation_matrix = np.identity(4)
    translation_matrix[2, 3] = distance
    new_pose = new_pose @ translation_matrix
    return new_pose


def rotate_camera_x(current_pose, theta_degrees):
    """
    Rotate the camera around the X-axis by a gven angle in degrees.
    """
    new_pose = current_pose.copy()
    theta_radians = np.radians(theta_degrees)
    rotation_matrix = np.array([[1, 0, 0, 0],
                                [0, np.cos(theta_radians), -
                                 np.sin(theta_radians), 0],
                                [0, np.sin(theta_radians),
                                 np.cos(theta_radians), 0],
                                [0, 0, 0, 1]])
    new_pose = new_pose @ rotation_matrix
    return new_pose


def rotate_camera_y(current_pose, theta_degrees):
    """
    Rotate the camera around the Y-axis by a gven angle in degrees.
    """
    new_pose = current_pose.copy()
    theta_radians = np.radians(theta_degrees)
    rotation_matrix = np.array([[np.cos(theta_radians), 0, np.sin(theta_radians), 0],
                                [0, 1, 0, 0],
                                [-np.sin(theta_radians), 0,
                                 np.cos(theta_radians), 0],
                                [0, 0, 0, 1]])
    new_pose = new_pose @ rotation_matrix
    return new_pose


def rotate_camera_z(current_pose, theta_degrees):
    """
    Rotate the camera around the Z-axis by a gven angle in degrees.
    """
    new_pose = current_pose.copy()
    theta_radians = np.radians(theta_degrees)
    rotation_matrix = np.array([[np.cos(theta_radians), -np.sin(theta_radians), 0, 0],
                                [np.sin(theta_radians), np.cos(
                                    theta_radians), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    new_pose = new_pose @ rotation_matrix
    return new_pose


def translate_semantic(points, mask, direction, step):
    delta = torch.zeros(3, device=points.device)
    if direction == 'X':
        delta[0] = step
    elif direction == 'Y':
        delta[1] = step
    elif direction == 'Z':
        delta[2] = step
    points[mask] = points[mask] + delta
    return points


def rotate_semantic(points, mask, direction, degrees):
    pts = points[mask]
    if pts.numel() == 0:
        return points
    center = pts.mean(dim=0, keepdim=True)
    rad = np.deg2rad(degrees)
    if direction == 'X':
        rot = torch.tensor([[1, 0, 0],
                            [0, np.cos(rad), -np.sin(rad)],
                            [0, np.sin(rad), np.cos(rad)]], device=points.device, dtype=points.dtype)
    elif direction == 'Y':
        rot = torch.tensor([[np.cos(rad), 0, np.sin(rad)],
                            [0, 1, 0],
                            [-np.sin(rad), 0, np.cos(rad)]], device=points.device, dtype=points.dtype)
    else:
        rot = torch.tensor([[np.cos(rad), -np.sin(rad), 0],
                            [np.sin(rad), np.cos(rad), 0],
                            [0, 0, 1]], device=points.device, dtype=points.dtype)
    shifted = pts - center
    rotated = (rot @ shifted.T).T + center
    points[mask] = rotated
    return points


def print_camera_pose(extrinsic_matrix):
    """
    Prints the x, y, and z axes from a camera's extrinsic matrix.

    Parameters:
    extrinsic_matrix (numpy array): A 4x4 extrinsic matrix of the camera.
    """

    # Check if the matrix is 4x4
    if extrinsic_matrix.shape != (4, 4):
        raise ValueError("Extrinsic matrix must be a 4x4 matrix.")

    # Extracting the rotation matrix (top-left 3x3)
    rotation_matrix = extrinsic_matrix[:3, :3]

    # The columns of the rotation matrix are the x, y, and z axes
    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]
    z_axis = rotation_matrix[:, 2]

    print(f"Camera position: X = {x_axis}, Y = {y_axis}, Z = {z_axis}")
    print(f"Camera extrinsic: {extrinsic_matrix}")


def visualize(scene_path, cfg):
    # Load Scene Data
    w2c, k = load_camera(cfg, scene_path)

    if 'load_semantics' in cfg:
        load_semantics = cfg['load_semantics']
    else:
        load_semantics = False

    scene_data, scene_depth_data, scene_semantic_data, all_w2cs, semantic_ids = load_scene_data(
        scene_path, w2c, k, load_semantics=load_semantics)
    if semantic_ids is not None:
        semantic_ids = semantic_ids.long()
    if load_semantics and scene_semantic_data is None:
        print("Semantic data not found in params; disabling semantic visualization.")
        load_semantics = False

    # Points to keep
    render_mask = torch.ones(
        scene_data['means3D'].shape[0], dtype=torch.bool).cuda()

    # vis.create_window()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    im, depth, sil = render(
        w2c, k, scene_data, cfg, render_mask,
        use_semantics=cfg.get('use_one_hot_semantics', False))
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    if cfg['visualize_cams']:
        # Initialize Estimated Camera Frustums
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(
                np.array(cam_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])

        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(line_colormap(
                (line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    # Initialize View Control

    # Adjust this factor to control the FOV. Less than 1.0 will increase FOV.
    focal_length_scale_factor = 0.8
    k[0, 0] *= focal_length_scale_factor
    k[1, 1] *= focal_length_scale_factor
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if cfg['offset_first_viz_cam']:
        view_w2c = w2c
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = w2c

    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(
        cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False
    render_options.background_color = [0.0, 0.0, 0.0]

    render_mode = cfg['render_mode']
    delta_trans = 0.2
    delta_rotate = 2.5
    delta_sem_trans = 0.05
    delta_sem_rotate = 5.0
    set_camera_w2c = False
    semantic_warned = False

    # Interactive Rendering
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        # Check for commands from Tkinter
        while not command_queue.empty():
            msg = command_queue.get()
            print(msg)
            if msg['type'] == Operation.SWITCH_MODE:
                render_mode = msg['payload']['mode']
            elif msg['type'] == Operation.APPLY_MASK:
                if not load_semantics or semantic_ids is None:
                    print("Semantic mask command ignored: semantic data not loaded.")
                    continue
                input_semantic_id = int(msg['payload']['semantic_id'])
                if input_semantic_id == -2:  # reset
                    render_mask = torch.ones(
                        scene_data['means3D'].shape[0], dtype=torch.bool).cuda()
                elif input_semantic_id == -1:  # no action
                    continue
                else:
                    to_keep = msg['payload']['to_keep']
                    render_mask[semantic_ids.squeeze(
                    ) == input_semantic_id] = bool(to_keep)
            elif msg['type'] == Operation.CAM_TRANS:
                set_camera_w2c = True
                if msg['payload']['direction'] == 'X':
                    w2c = move_camera_x(
                        w2c, msg['payload']['factor'] * delta_trans)
                elif msg['payload']['direction'] == 'Y':
                    w2c = move_camera_y(
                        w2c, msg['payload']['factor'] * delta_trans)
                elif msg['payload']['direction'] == 'Z':
                    w2c = move_camera_z(
                        w2c, msg['payload']['factor'] * delta_trans)
            elif msg['type'] == Operation.CAM_ROTATE:
                set_camera_w2c = True
                if msg['payload']['direction'] == 'X':
                    w2c = rotate_camera_x(
                        w2c, msg['payload']['factor'] * delta_rotate)
                elif msg['payload']['direction'] == 'Y':
                    w2c = rotate_camera_y(
                        w2c, msg['payload']['factor'] * delta_rotate)
                elif msg['payload']['direction'] == 'Z':
                    w2c = rotate_camera_z(
                        w2c, msg['payload']['factor'] * delta_rotate)
            elif msg['type'] in (Operation.SEM_TRANS, Operation.SEM_ROTATE, Operation.SEM_DELETE):
                if not load_semantics or semantic_ids is None:
                    print("Semantic edit command ignored: semantic data not loaded.")
                    continue
                target_id = int(msg['payload'].get('semantic_id', -1))
                target_mask = semantic_ids.squeeze() == target_id
                if not torch.any(target_mask):
                    print(f"No points found for semantic id {target_id}")
                    continue
                if msg['type'] == Operation.SEM_DELETE:
                    render_mask[target_mask] = False
                elif msg['type'] == Operation.SEM_TRANS:
                    direction = msg['payload']['direction']
                    step = msg['payload']['factor'] * delta_sem_trans
                    scene_data['means3D'] = translate_semantic(
                        scene_data['means3D'], target_mask, direction, step)
                    if scene_semantic_data is not None:
                        scene_semantic_data['means3D'] = translate_semantic(
                            scene_semantic_data['means3D'], target_mask, direction, step)
                    scene_depth_data['means3D'] = translate_semantic(
                        scene_depth_data['means3D'], target_mask, direction, step)
                elif msg['type'] == Operation.SEM_ROTATE:
                    direction = msg['payload']['direction']
                    deg = msg['payload']['factor'] * delta_sem_rotate
                    scene_data['means3D'] = rotate_semantic(
                        scene_data['means3D'], target_mask, direction, deg)
                    if scene_semantic_data is not None:
                        scene_semantic_data['means3D'] = rotate_semantic(
                            scene_semantic_data['means3D'], target_mask, direction, deg)
                    scene_depth_data['means3D'] = rotate_semantic(
                        scene_depth_data['means3D'], target_mask, direction, deg)

        if render_mode == 'centers':
            pts = o3d.utility.Vector3dVector(
                scene_data['means3D'][render_mask].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(
                scene_data['colors_precomp'][render_mask].contiguous().double().cpu().numpy())
        elif render_mode == 'semantic_color':
            if scene_semantic_data is None:
                if not semantic_warned:
                    print(
                        "Semantic visualization requested but semantic data missing; falling back to color mode.")
                    semantic_warned = True
                im, depth, sil = render(
                    w2c, k, scene_data, cfg, render_mask,
                    use_semantics=cfg.get('use_one_hot_semantics', False))
                if cfg['show_sil']:
                    im = (1-sil).repeat(3, 1, 1)
                pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
            else:
                seg, depth, sil = render(
                    w2c, k, scene_semantic_data, cfg, render_mask,
                    use_semantics=cfg.get('use_one_hot_semantics', False))
                pts, cols = rgbd2pcd(seg, depth, w2c, k, cfg)
        else:
            im, depth, sil = render(
                w2c, k, scene_data, cfg, render_mask,
                use_semantics=cfg.get('use_one_hot_semantics', False))
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

        if set_camera_w2c:
            cam_params.extrinsic = w2c
            print_camera_pose(w2c)
            view_control.convert_from_pinhole_camera_parameters(
                cam_params, allow_arbitrary=True)
            set_camera_w2c = False

    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]

    print(scene_path)
    viz_cfg = experiment.config["viz"]

    available_semantic_ids = extract_semantic_ids_from_params(
        scene_path) if viz_cfg.get('load_semantics', False) else set()
    if available_semantic_ids:
        print(f"Available semantic ids: {sorted(available_semantic_ids)}")
    elif viz_cfg.get('load_semantics', False):
        print("No semantic ids found in params; semantic mask operations will be disabled.")

    # Start the Open3D visualizer in a separate thread
    thread = Thread(target=visualize, args=(scene_path, viz_cfg))
    thread.daemon = True
    thread.start()

    # Create the Tkinter window
    root = tk.Tk()

    # Create the control panel
    control_panel = ControlPanel(root, command_queue, viz_cfg, width=20, height=200,
                                 semantic_ids=available_semantic_ids)

    # Run the application
    root.mainloop()

import numpy as np
import cv2
from rgbd_mocap.markers.marker_set import MarkerSet
from rgbd_mocap.processing.config import load_json


class Projector:
    def __init__(self, config):
        self.config = load_json(config)
        self.marker_sets = self._init_marker_set()

    def project_real(self, mapping_real_virt, virt_pose, virt_name_list):
        proj_markers = []
        markers_names = sum([ms.get_markers_names() for ms in self.marker_sets], [])
        for real_name, virt_names in mapping_real_virt:
            idx_real = markers_names.index(real_name)
            idx_virt = [virt_name_list.index(virt_name) for virt_name in virt_names if virt_name in virt_name_list]
            pose_virt = [virt_pose[idx] for idx in idx_virt]
            basis = self._get_2d_basis(pose_virt[0], pose_virt[1], [0, 0])
            real_proj = self._project_from_basis_to_img(self.local_real[real_name], basis, self._from_2d_to_3d(pose_virt[0]))
            proj_markers.append(real_proj)
        return proj_markers
    

    def project_real_in_local(self, mapping_real_virt):
        markers_names, markers_pose = self.get_pose_global()
        local_real = {}
        for real_name, virt_names in mapping_real_virt:
            idx_real = markers_names.index(real_name)
            idx_virt = [markers_names.index(virt_name) for virt_name in virt_names if virt_name in markers_names]
            pose_real = markers_pose[idx_real]
            pose_virt = [markers_pose[idx] for idx in idx_virt]
            basis = self._get_2d_basis(pose_virt[0], pose_virt[1], [0, 0])
            loc_tmp = self._project_from_img_to_basis(
                self._from_2d_to_3d(pose_real), basis, self._from_2d_to_3d(pose_virt[0])
            )
            local_real[real_name] = loc_tmp
        self.local_real = local_real

    def _init_marker_set(self):
        set_names = []
        off_sets = []
        marker_names = []
        base_positions = []
        for i in range(len(self.config["crops"])):
            set_names.append(self.config["crops"][i]["name"])
            off_sets.append(self.config["crops"][i]["area"])
            marker_name = []
            base_position = []
            for j in range(len(self.config["crops"][i]["markers"])):
                marker_name.append(self.config["crops"][i]["markers"][j]["name"])
                base_position.append(np.array(
                    [
                        self.config["crops"][i]["markers"][j]["pos"][1],
                        self.config["crops"][i]["markers"][j]["pos"][0],
                    ])
                )
            marker_names.append(marker_name)
            base_positions.append(base_position)

        marker_sets: list[MarkerSet] = []
        for i in range(len(set_names)):
            marker_set = MarkerSet(set_names[i], marker_names[i], False, downsample_ratio=1)
            marker_set.set_markers_pos(base_positions[i])
            marker_set.set_offset_pos(off_sets[i][:2])
            marker_sets.append(marker_set)
        
        return marker_sets

    def get_pose_global(self):
        pose = sum([ms.get_markers_global_pos() for ms in self.marker_sets], [])
        names = sum([ms.get_markers_names() for ms in self.marker_sets], [])
        return names, pose

    @staticmethod
    def _get_2d_basis(a, b, c):
        vect_0 = np.zeros(3)
        vect_0[:2] = b - a
        vect_1 = np.zeros(3)
        vect_1[:2] = c - b
        vect_0 = vect_0 / np.linalg.norm(vect_0)
        vect_1 = vect_1 / np.linalg.norm(vect_1)
        perp = np.cross(vect_0, vect_1)
        orthogonal_1 = np.cross(vect_0, perp)
        return np.array([vect_0, orthogonal_1])

    @staticmethod
    def _project_from_img_to_basis(coordinate, basis, origin):
        vect_0, orthogonal_1 = basis
        coord_0 = np.dot(coordinate - origin, vect_0)
        coord_1 = np.dot(coordinate - origin, orthogonal_1)
        return coord_0, coord_1

    @staticmethod
    def _project_from_basis_to_img(point, basis, origin):
        coord_0, coord_1 = point
        vect_0, orthogonal_1 = basis
        coordinate = coord_0 * vect_0 + coord_1 * orthogonal_1 + origin
        return coordinate

    @staticmethod
    def _from_2d_to_3d(points):
        point_3d = np.zeros((3))
        point_3d[:2] = points
        return point_3d

def inpaint(frame, center_list, radius):
    for center in center_list:
        mask = np.zeros_like(frame)
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        frame = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
    return frame

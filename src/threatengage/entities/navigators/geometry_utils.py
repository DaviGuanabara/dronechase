import numpy as np


class GeometryUtils:
    @staticmethod
    def is_point_inside_cone(point, apex_position, base_center_position, degrees=60):
        if apex_position is None or base_center_position is None:
            raise ValueError("apex_position and base_center_position must not be None")

        apex_to_base = np.array(base_center_position) - np.array(apex_position)
        apex_to_point = np.array(point) - np.array(apex_position)

        if np.linalg.norm(apex_to_point) > np.linalg.norm(apex_to_base):
            return False

        max_degree_between_vectors = degrees / 2
        degree_between_vectors = GeometryUtils.degrees_between_vectors(
            apex_to_point, apex_to_base
        )
        return degree_between_vectors <= max_degree_between_vectors

    @staticmethod
    def degrees_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_angle = dot_product / (norm_v1 * norm_v2)
        angle_radians = np.arccos(cos_angle)
        return np.degrees(angle_radians)

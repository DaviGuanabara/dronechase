import pytest
from threatengage.modules.quadcoters.components.sensors.lidar import LiDAR, CoordinateConverter, Channels, QuadcopterType  # adjust the import path accordingly
import numpy as np
import pybullet as p

class TestLiDAR:

    def setup_method(self):
        """Setup method called before every test case."""
        self.lidar = LiDAR(parent_id=1, client_id=1)

    def test_spherical_to_cartesian_conversion(self):
        spherical = np.array([1, np.pi/4, np.pi/4])  # Radius=1, Theta=45deg, Phi=45deg
        cartesian = CoordinateConverter.spherical_to_cartesian(spherical)
        assert np.isclose(cartesian, np.array([0.5, 0.5, np.sqrt(2)/2])).all()

    def test_cartesian_to_spherical_conversion(self):
        cartesian = np.array([0.5, 0.5, np.sqrt(2)/2])
        spherical = CoordinateConverter.cartesian_to_spherical(cartesian)
        assert np.isclose(spherical, np.array([1, np.pi/4, np.pi/4])).all()

    def test_gen_sphere(self):
        assert self.lidar.get_sphere().shape == (len(Channels), self.lidar.n_theta_points, self.lidar.n_phi_points)

    def test_get_flag(self):
        assert self.lidar._get_flag(QuadcopterType.LOYALWINGMAN) == 0.3

    def test_data_shape(self):
        shape = self.lidar.get_data_shape()
        assert shape == (self.lidar.n_channels, self.lidar.n_theta_points, self.lidar.n_phi_points)
        
    def test_buffer_publisher_inertial_data(self):
        lidar_sensor = LiDAR(parent_id=1, client_id=0, radius=2, resolution=1, debug=False)
        assert len(lidar_sensor.buffer) == 0, "Buffer should be empty at the beginning"
        
        parent_inertial_data = {
            'type': QuadcopterType.LOYALWINGMAN,
            'position': np.array([0, 0, 0]),
            'attitude': np.array([0, 0, 0]),
            'quaternion': p.getQuaternionFromEuler(np.array([0, 0, 0]))
        }
        parent_publisher_id = 1
        
        sample_inertial_data = {
            'type': QuadcopterType.LOYALWINGMAN,
            'position': np.array([1, 0, 0])
        }
        sample_publisher_id = 2
        
        lidar_sensor.buffer_inertial_data(parent_inertial_data, parent_publisher_id)
        assert hasattr(lidar_sensor, 'parent_inertia'), "parent inertial data was not set after adding a new inertial data"
        lidar_sensor.buffer_inertial_data(sample_inertial_data, sample_publisher_id)
        assert len(lidar_sensor.buffer) > 0, "Buffer should be populated after adding a new inertial data"
        
    def test_lidar_sensor(self):
        # Instantiating the LiDAR sensor
        lidar_sensor = LiDAR(parent_id=1, client_id=0,radius=2, resolution=1, debug=False)
        
        # Generating some sample inertial data
        parent_inertial_data = {
            'type': QuadcopterType.LOYALWINGMAN,
            'position': np.array([0, 0, 0]),
            'attitude': np.array([0, 0, 0]),
            'quaternion': p.getQuaternionFromEuler(np.array([0, 0, 0]))
        }
        parent_publisher_id = 1
        
        sample_inertial_data = {
            'type': QuadcopterType.LOYALWINGMAN,
            'position': np.array([1, 0, 0])
        }
        sample_publisher_id = 2

        # Step 1: Buffer the inertial data
        lidar_sensor.buffer_inertial_data(parent_inertial_data, parent_publisher_id)
        lidar_sensor.buffer_inertial_data(sample_inertial_data, sample_publisher_id)
        
        # Step 2: Update the sensor's internal state/data
        lidar_sensor.update_data()

        # Step 3: Read the data from the sensor
        data = lidar_sensor.read_data()
        
        # For simplicity, just print the data (you can add asserts or other tests if required)
        print(data)


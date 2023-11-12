# 1. Quadcopter Class Updates
## 1.1 Retrieve Dimensions at Initialization
In the __init__ method of the Quadcopter class, retrieve the dimensions of the quadcopter using PyBullet's getVisualShapeData method.
Store the dimensions as an attribute of the Quadcopter class.

```python

self.dimensions = p.getVisualShapeData(self.id)[0][3]
```

## 1.2 Update IMU Inertial Data with Dimensions
In the update_state method (or wherever the IMU data is being updated), include the dimensions in the inertial data message being published.

```python
inertial_data = self.imu.read_data()
inertial_data['dimensions'] = self.dimensions.tolist()
self._publish_inertial_data(inertial_data)
```

# 2. LiDAR Class Updates

The message received has to be processed in a way to extract the extremety points of the quadcopter object.
These extremity points have to be added to the LiDAR buffer for processing.



## 2.1. Buffer Dimensions in LiDAR Class
In the LiDAR class, extend the buffer_inertial_data method to handle the dimensions if they are present in the message.

## 2.2. Calculate Intermediate Points between Extremities Once Updated is Executed

When update is executed, we need first extract the extremety points of the quadcopter object, then calculate the intermediate points between them, finishing by adding then to the matrix of points to be processed by the LiDAR sensor.

This could be achieved using interpolation or a similar technique.

```python
def calculate_extremity_points(self, position, dimensions):
    half_dimensions = dimensions / 2
    extremity_points = {
        "top": position + [0, 0, half_dimensions[2]],
        "bottom": position - [0, 0, half_dimensions[2]],
        "left": position - [half_dimensions[0], 0, 0],
        "right": position + [half_dimensions[0], 0, 0],
        "front": position + [0, half_dimensions[1], 0],
        "back": position - [0, half_dimensions[1], 0],
    }
    return extremity_points

```

```python
def calculate_intermediate_points(self, point1: np.ndarray, point2: np.ndarray) -> List[np.ndarray]:
    # Implement the logic to calculate intermediate points
    # ...
    return intermediate_points
```

## 2.3. Add Intermediate Points to LiDAR Buffer
Use the calculate_intermediate_points method to generate intermediate points between the extremity points.
Add these intermediate points to the LiDAR buffer for processing.



## 2.4. Update LiDAR Data with Intermediate Points
In the update_data method of the LiDAR class, ensure that the intermediate points are processed along with other data points.
This might involve adding the intermediate points to the LiDAR sphere or handling them in a specific manner.



# 3. Documentation
Document the process and the methods involved in the codebase for future reference and clarity.
By following these steps, you should be able to calculate and utilize the extremity points of the quadcopter object within the LiDAR class.
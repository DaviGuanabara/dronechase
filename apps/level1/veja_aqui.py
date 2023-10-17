import numpy as np

# Step 1: Generate random vectors for velocity and direction
velocity = np.array([1, 1, 1])
direction = np.array([1, -1, -1])

# Make sure that the direction vector is normalized (has a magnitude of 1)
direction = direction / np.linalg.norm(direction)

print("Velocity:", velocity)
print("Direction:", direction)

# Step 2: Compute the dot product
bonus = 0  # Assuming bonus starts at 0
bonus += np.dot(velocity, direction)
print("Bonus:", bonus)

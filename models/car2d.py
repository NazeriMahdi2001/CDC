import numpy as np

class Robot2D:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

    def set_state(self, x, y):
        self.x = x
        self.y = y

        return self

    def get_state(self):
        return self.x, self.y
    
    def update_dynamics(self, control):

        new_x = self.x + 10 * control[0] * np.cos(control[1])
        new_y = self.y + 10 * control[0] * np.sin(control[1])

        self.x = new_x
        self.y = new_y
        
        return self.get_state()
    
    def vectorized_dynamics(self, state, control):
        new_x = state[:, 0] + 10 * control[:, 0] * np.cos(control[:, 1])
        new_y = state[:, 1] + 10 * control[:, 0] * np.sin(control[:, 1])

        return np.stack([new_x, new_y], axis=1)

    def L_X(self, state, control):
        # Partial derivatives of the dynamics equations with respect to system state
        d_1_d_1 = 1
        d_1_d_2 = 0
        d_2_d_1 = 0
        d_2_d_2 = 1

        max_jacobian_matrix = np.array([
            [d_1_d_1, d_1_d_2],
            [d_2_d_1, d_2_d_2]
        ])

        return np.linalg.matrix_power(max_jacobian_matrix, 1)

    def L_U(self, state, control):
        # Partial derivatives of the dynamics equations with respect to contorl input
        d_1_d_1 = 10
        d_1_d_2 = 1
        d_2_d_1 = 10
        d_2_d_2 = 1

        max_jacobian_matrix = np.array([
            [d_1_d_1, d_1_d_2],
            [d_2_d_1, d_2_d_2]
        ])

        return np.linalg.matrix_power(max_jacobian_matrix, 1)
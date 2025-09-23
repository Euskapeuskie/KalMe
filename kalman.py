from enum import Enum
from generate_measurements import Measurements_2D, Systems_2D
import matplotlib.pyplot as plt
import numpy as np


class KalmanModel(Enum):
    CONSTANT_VELOCITY = 1
    CONSTANT_ACCELERATION = 2


class KalmanFilter:
    def __init__(self, dt, process_variance, measurement_variance, dimensions=1, model=KalmanModel.CONSTANT_VELOCITY):
        # Time step
        self.dt = dt
        self.model = model
        self.dimensions = dimensions

        # Initialize matrices based on the selected model
        match model:
            case KalmanModel.CONSTANT_VELOCITY:
                self.n_state_dim = dimensions * 2  # [x0_position, x0_velocity, x1_position, x1_velocity]
                self._init_constant_velocity(dt, process_variance, measurement_variance)
            case KalmanModel.CONSTANT_ACCELERATION:
                self.n_state_dim = dimensions * 3  # [x0_position, x0_velocity, x0_acceleration, x1_position, x1_velocity, x1_acceleration]
                self._init_constant_acceleration(dt, process_variance, measurement_variance)
            case _:
                raise ValueError("Unsupported Kalman Model")
            
        # Initial state vector
        self.x = np.zeros((self.n_state_dim, 1))

        # Initial covariance matrix - no covariance between positions and velocities
        self.P = np.eye(self.n_state_dim)

        # Initial Measurement noise covariance
        self.R = np.array([[measurement_variance]]) * np.eye(dimensions)


    def _init_constant_velocity(self, dt, process_variance, measurement_variance):

        # State transition matrix
        # x0 = x0 + v0*dt
        # v0 = v0
        # x1 = x1 + v1*dt
        # v1 = v1
        F1D = np.array([[1, dt],
                        [0, 1]])    
        self.F = self._build_block_diag(F1D, self.dimensions)

        # Measurement matrix
        # We only measure position in x0 and x1
        H1D = np.array([[1, 0]])
        self.H = self._build_block_diag(H1D, self.dimensions)

        # Process noise covariance
        Q1D = process_variance * np.array([[dt**4 / 4, dt**3 / 2],
                                           [dt**3 / 2, dt**2]])
        self.Q = self._build_block_diag(Q1D, self.dimensions)

    
    def _init_constant_acceleration(self, dt, process_variance, measurement_variance):

        # State transition matrix
        # x0 = x0 + v0*dt + 0.5*a0*dt^2
        # v0 = v0 + a0*dt
        # a0 = a0
        # x1 = x1 + v1*dt + 0.5*a1*dt^2
        # v1 = v1 + a1*dt
        # a1 = a1
        F1D = np.array([[1, dt, dt**2/2],
                        [0, 1, dt],
                        [0, 0, 1]])
        self.F = self._build_block_diag(F1D, self.dimensions)

        # Measurement matrix
        # We only measure position in x0 and x1
        H1D = np.array([[1, 0, 0]])
        self.H = self._build_block_diag(H1D, self.dimensions)

        # Process noise covariance
        Q1D = process_variance * np.array([[dt**5/20, dt**4/8, dt**3/6],
                                           [dt**4/8, dt**3/3, dt**2/2],
                                           [dt**3/6, dt**2/2, dt]])
        self.Q = self._build_block_diag(Q1D, self.dimensions)
    

    def _build_block_diag(self, A, n):
        return np.block([[A if i == j else np.zeros_like(A) for j in range(n)] for i in range(n)])
        

    def predict(self):
        # Predict the next state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        z: measurement vector [x0_position, x1_position]
        """

        # Transform z to a column vector
        # [x0_position, x1_position] -> [[x0_position], [x1_position]]
        match self.dimensions:
            case 1:
                z = z[-1].reshape(-1, 1)  # only last element is position in 1D case
            case 2:
                z  = z.reshape(-1, 1)
        # Measurement residual
        y = z - (self.H @ self.x)

        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate and covariance matrix
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        match self.model:
            case KalmanModel.CONSTANT_VELOCITY:
                return self.x.flatten()[[2*i for i in range(self.dimensions)]]  # return positions only
            case KalmanModel.CONSTANT_ACCELERATION: 
                return self.x.flatten()[[3*i for i in range(self.dimensions)]]  # return positions only
            case _:
                return self.x.flatten()
    
    def filter_batch(self, measurements):
        """
        measurements: array of measurement vectors [[x0_position, x1_position], ...]
        Returns: array of filtered state vectors [[x0_position, x0_velocity, x1_position, x1_velocity], ...]
        """
        filtered_states = []
        for z in measurements:
            self.predict()
            self.update(z)
            filtered_states.append(self.get_state())
        return np.array(filtered_states)
    


if __name__ == "__main__":
    # Example usage
    dt = 0.05  # Time step
    process_variance = 0  # Process noise variance
    measurement_variance = 0.05  # Measurement noise variance
    model = KalmanModel.CONSTANT_VELOCITY

    # Create a Kalman Filter instance for constant velocity model
    kf = KalmanFilter(dt, process_variance, measurement_variance, model=model)
    
    # generate some noisy measurements
    zs = Measurements_2D(dt, measurement_variance).gen_measurements(type=Systems_2D.CIRCLE)

    for z in zs:
        print(z)
        kf.predict()
        kf.update(np.array([[z]]))
        x = kf.get_state()
        # filtered data
        plt.plot(x[0], x[1], 'ro', color='blue')
        # raw data
        plt.plot(z[0], z[1], 'ro-', color="red")
        
    plt.show()
import numpy as np
from enum import Enum

class Systems_2D(Enum):
    LINEAR = "linear"
    CIRCLE = "circle"
    FIGURE_8 = "figure_8"


class Systems_1D(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SINUSOIDAL = "sinusoidal"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_LINEARIZED = "exponential_linearized"


class Measurements_2D:
    def __init__(self, dt, measurement_variance):
        self.dt = dt
        self.measurement_variance = measurement_variance

    def gen_measurements(self, type=Systems_2D.CIRCLE):
        match type:
            case Systems_2D.CIRCLE:
                return self._circle()
            case Systems_2D.LINEAR:
                return self._linear()
            case Systems_2D.FIGURE_8:
                return self._figure_8()
            case _:
                raise ValueError("Unsupported measurement type")
    
    def _circle(self):
        """
        Generates noisy measurements in a circular pattern \n
        Returns: np.array(x), np.array(y) scaled from 0 to 1
        """
        t = np.arange(0, 2*np.pi, self.dt)
        x = 0.5 + 0.5 * np.cos(t) + np.random.normal(0, self.measurement_variance, len(t))
        y = 0.5 + 0.5 * np.sin(t) + np.random.normal(0, self.measurement_variance, len(t))

        zs = np.reshape(np.array([x, y]).T, (-1, 2))
        return zs
    
    def _linear(self):
        """
        Generates noisy measurements in a linear pattern \n
        Returns: np.array(x), np.array(y) scaled from 0 to 1
        """
        t = np.arange(0, 1, self.dt)
        x = t + np.random.normal(0, self.measurement_variance, len(t))
        y = t + np.random.normal(0, self.measurement_variance, len(t))

        zs = np.reshape(np.array([x, y]).T, (-1, 2))
        return zs
    
    def _figure_8(self):
        """
        Generates noisy measurements in a figure 8 pattern \n
        Returns: np.array(x), np.array(y) scaled from 0 to 1
        """
        t = np.arange(0, 2*np.pi, self.dt)
        x = 0.5 + 0.5 * np.sin(t) + np.random.normal(0, self.measurement_variance, len(t))
        y = 0.5 + 0.5 * np.sin(t) * np.cos(t) + np.random.normal(0, self.measurement_variance, len(t))

        zs = np.reshape(np.array([x, y]).T, (-1, 2))
        return zs


class Measurements_1D:
    def __init__(self, dt, measurement_variance):
        self.dt = dt
        self.measurement_variance = measurement_variance

    def gen_measurements(self, type=Systems_1D.LINEAR):
        match type:
            case Systems_1D.LINEAR:
                return self._linear()
            case Systems_1D.QUADRATIC:
                return self._quadratic()
            case Systems_1D.SINUSOIDAL:
                return self._sinusoidal()
            case Systems_1D.EXPONENTIAL:
                return self._exponential()
            case Systems_1D.EXPONENTIAL_LINEARIZED:
                return self._exponential_linearized()
            case _:
                raise ValueError("Unsupported measurement type")
    
    def _linear(self):
        """
        Generates noisy measurements in a linear pattern \n
        Returns: np.array(x) scaled from 0 to 1
        """
        t = np.arange(0, 1, self.dt)
        x = t + np.random.normal(0, self.measurement_variance, len(t))

        zs = np.reshape(np.array([t, x]).T, (-1, 2))
        return zs
    
    def _quadratic(self):
        """
        Generates noisy measurements in a quadratic pattern \n
        Returns: np.array(x) scaled from 0 to 1
        """
        t = np.arange(0, 1, self.dt)
        x = t**2 + np.random.normal(0, self.measurement_variance, len(t))

        zs = np.reshape(np.array([t, x]).T, (-1, 2))
        return zs
    
    def _sinusoidal(self):
        """
        Generates noisy measurements in a sinusoidal pattern \n
        Returns: np.array(x) scaled from 0 to 1
        """
        t = np.arange(0, 2*np.pi, self.dt)
        x = 0.5 + 0.5 * np.sin(t) + np.random.normal(0, self.measurement_variance, len(t))

        zs = np.reshape(np.array([t, x]).T, (-1, 2))
        return zs
    
    def _exponential(self):
        """
        Generates noisy measurements in an exponential pattern \n
        Returns: np.array(x) scaled from 0 to 1
        """
        t = np.arange(0, 5, self.dt)
        x = np.exp(t) * (1 + np.random.normal(0, self.measurement_variance, len(t)))

        zs = np.reshape(np.array([t, x]).T, (-1, 2))
        return zs
    
    def _exponential_linearized(self):
        """
        Generates noisy measurements in an exponential pattern that has been linearized via log transform \n
        Returns: np.array(x) scaled from 0 to 1
        """
        t = np.arange(0, 5, self.dt)
        x = np.exp(t) * (1 + np.random.normal(0, self.measurement_variance, len(t)))
        x = np.log(x)  # linearize the exponential data

        zs = np.reshape(np.array([t, x]).T, (-1, 2))
        return zs
    
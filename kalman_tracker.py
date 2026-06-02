import numpy as np

class KalmanTracker:
    """
    1D/2D/4D Kalman filter for smooth tracking.
    Supports arbitrary state dimensions with position and velocity.
    """

    def __init__(self, state_dim=2, process_noise=0.01, measurement_noise=1.0, dt=0.033):
        """
        Initialize Kalman filter.
        state_dim: dimension of state (e.g., 2 for (x,y), 4 for (x1,y1,x2,y2))
        process_noise: process noise (Q)
        measurement_noise: measurement noise (R)
        dt: time step (default ~30ms for 30fps)
        """
        self.dim = state_dim
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State: [x, vx, y, vy, ...] for 2D, [x1, y1, x2, y2] for 4D box
        # For simplicity: treat each dimension independently
        self.x = np.zeros(state_dim)  # Position
        self.v = np.zeros(state_dim)  # Velocity

        # Covariance
        self.P = np.eye(state_dim) * 10.0

    def predict(self):
        """Predict next state."""
        # x(k+1) = x(k) + v(k) * dt
        self.x = self.x + self.v * self.dt

        # Update covariance: P = P + Q
        Q = np.eye(self.dim) * self.process_noise
        self.P = self.P + Q

        return self.x.copy()

    def update(self, measurement):
        """
        Update state with measurement.
        measurement: np.array of same dimension as state
        """
        if measurement is None:
            return self.x.copy()

        measurement = np.array(measurement, dtype=float)

        # Innovation
        y = measurement - self.x

        # Innovation covariance
        R = np.eye(self.dim) * self.measurement_noise
        S = self.P + R

        # Kalman gain (diagonal case simplified)
        diag_S = np.diag(S)
        K = np.diag(self.P) / (diag_S + 1e-6)
        K = np.clip(K, 0, 1)

        # Update state
        self.x = self.x + K * y

        # Update velocity estimate (for smoothing)
        self.v = (self.v + K * y / self.dt) * 0.8 + self.v * 0.2

        # Update covariance
        K_matrix = np.diag(K)
        self.P = (np.eye(self.dim) - K_matrix) @ self.P

        return self.x.copy()

    def get_state(self):
        """Get current state."""
        return self.x.copy()

    def initialize(self, measurement):
        """Initialize state with measurement."""
        self.x = np.array(measurement, dtype=float)
        self.P = np.eye(self.dim) * 10.0


class TargetTracker:
    """
    Combined tracker for both target point and target object box.
    """

    def __init__(self, point_process_noise=0.05, point_measurement_noise=2.0,
                 box_process_noise=0.02, box_measurement_noise=5.0):
        """
        Initialize target tracker with separate Kalman filters for point and box.
        """
        # 2D point tracker (x, y)
        self.point_tracker = KalmanTracker(
            state_dim=2,
            process_noise=point_process_noise,
            measurement_noise=point_measurement_noise
        )

        # 4D box tracker (x1, y1, x2, y2)
        self.box_tracker = KalmanTracker(
            state_dim=4,
            process_noise=box_process_noise,
            measurement_noise=box_measurement_noise
        )

        self.is_initialized = False

    def initialize(self, target_point, target_box):
        """
        Initialize trackers with initial measurements.
        target_point: (x, y)
        target_box: (x1, y1, x2, y2)
        """
        self.point_tracker.x = np.array(target_point, dtype=float)
        self.box_tracker.x = np.array(target_box, dtype=float)
        self.is_initialized = True

    def predict(self):
        """Predict next state for both point and box."""
        point_pred = self.point_tracker.predict()
        box_pred = self.box_tracker.predict()
        return point_pred, box_pred

    def update(self, target_point, target_box):
        """
        Update trackers with new measurements.
        Returns: (tracked_point, tracked_box)
        """
        if not self.is_initialized:
            self.initialize(target_point, target_box)

        point_tracked = self.point_tracker.update(target_point)
        box_tracked = self.box_tracker.update(target_box)

        return point_tracked, box_tracked

    def track(self, target_point, target_box):
        """
        Predict and update in one call.
        Returns: (tracked_point, tracked_box)
        """
        self.predict()
        return self.update(target_point, target_box)

    def reset(self):
        """Reset tracker state."""
        self.is_initialized = False
        self.point_tracker = KalmanTracker(state_dim=2)
        self.box_tracker = KalmanTracker(state_dim=4)

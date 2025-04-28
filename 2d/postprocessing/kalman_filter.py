class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_estimate=0.0, initial_error=1.0):
        self.process_variance = process_variance      # Q: Process (model) noise
        self.measurement_variance = measurement_variance  # R: Measurement noise
        self.estimate = initial_estimate              # x̂: Initial estimate
        self.error = initial_error                    # P: Estimate error

    def update(self, measurement):
        # Prediction step
        self.error += self.process_variance

        # Kalman gain
        kalman_gain = self.error / (self.error + self.measurement_variance)

        # Correction step
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.error *= (1 - kalman_gain)

        return self.estimate

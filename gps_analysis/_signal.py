"""Kalman + RTS smoothing for GPS trajectories."""
import numpy as np


def kalman_smooth_track(gx, gy, t, R_pos=0.06, Q_accel=500.0):
    """Apply Kalman filter + RTS smoother to a single GPS track.

    Runs directly on the raw data (typically 10 Hz). Uses a
    constant-velocity model with fixed dt (median sample interval).

    Parameters
    ----------
    gx, gy : arrays  — GPS positions in grid units.
    t : array         — Time in minutes.
    R_pos : float     — GPS measurement noise variance (grid units²).
                        0.06 ≈ (2.4 m)² matches typical consumer GNSS.
    Q_accel : float   — Process noise (acceleration variance).
                        5.0 allows realistic sheep acceleration while
                        still smoothing GPS jitter.

    Returns
    -------
    gx_smooth, gy_smooth : arrays (same length as input).
    """
    n = len(gx)
    if n < 3:
        return gx.copy(), gy.copy()

    # Use median dt (handles minor irregular sampling)
    dt = float(np.median(np.diff(t[:min(200, n)])))
    if dt <= 0:
        dt = 1.0 / 600  # fallback 10 Hz

    # Constant-dt matrices
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                   [0, 0, 1, dt], [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
    R = np.eye(2) * R_pos
    q = Q_accel
    Q = np.array([
        [dt**3/3, dt**2/2, 0, 0],
        [dt**2/2, dt,      0, 0],
        [0, 0, dt**3/3, dt**2/2],
        [0, 0, dt**2/2, dt],
    ], dtype=float) * q

    # Forward Kalman filter
    mus = np.empty((n, 4))
    covs = np.empty((n, 4, 4))
    x = np.array([gx[0], 0.0, gy[0], 0.0])
    P = np.eye(4) * 0.1

    for i in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = np.linalg.solve(S.T, (P @ H.T).T).T
        z = np.array([gx[i], gy[i]])
        x = x + K @ (z - H @ x)
        P = P - K @ H @ P
        mus[i] = x
        covs[i] = P

    # RTS backward smoother
    for i in range(n - 2, -1, -1):
        Pp = F @ covs[i] @ F.T + Q
        C = np.linalg.solve(Pp.T, (covs[i] @ F.T).T).T
        mus[i] = mus[i] + C @ (mus[i + 1] - F @ mus[i])

    return mus[:, 0], mus[:, 2]

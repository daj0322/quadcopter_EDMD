import numpy as np
import matplotlib.pyplot as plt
from interception_env import (
    TargetConfig,
    InterceptorConfig,
    ScenarioConfig,
    simulate_open_loop_interception,
)

scenario_cfg = ScenarioConfig(
    dt=0.01,
    t_final=20.0,
    capture_radius=0.5,
)

target_cfg = TargetConfig(
    p0=np.array([2.0, 0.0, -1.0]),
    v=np.array([0.3, 0.0, 0.0]),
    head_start=2.0,
)

interceptor_cfg = InterceptorConfig(
    x0=np.zeros(12),
)

results = simulate_open_loop_interception(
    scenario_cfg=scenario_cfg,
    target_cfg=target_cfg,
    interceptor_cfg=interceptor_cfg,
)

print("Captured:", results["captured"])
print("Capture time:", results["capture_time"])
print("Minimum distance:", results["min_dist"])

t = results["t"]
x_i = results["x_i"]
p_t = results["p_t"]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_i[:, 0], x_i[:, 1], x_i[:, 2], label="Interceptor")
ax.plot(p_t[:, 0], p_t[:, 1], p_t[:, 2], label="Target")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Straight-line interception debug")
ax.legend()
plt.show()
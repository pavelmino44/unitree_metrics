from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Load TensorBoard scalars
# =========================

def load_scalar(run_path, tag):
    event_file = next(
        os.path.join(run_path, f)
        for f in os.listdir(run_path)
        if f.startswith("events.out")
    )

    ea = EventAccumulator(event_file)
    ea.Reload()

    events = ea.Scalars(tag)

    steps = [e.step for e in events]
    values = [e.value for e in events]

    return steps, values


# =========================
# Settings
# =========================

runs = [
    "logs/run_1",
    "logs/run_2",
    "logs/run_3",
    "logs/run_4",
    "logs/run_5",
]

SKIP_FIRST = 1000
DROP_LAST = 100

# зоны удаления (для визуализации)
reward_zones = [
    (8800, 10200),
    (13700, 15100),
    (15900, 17300),
    (26400, 27800),
    (29600, 29999),
]

episode_zones = [
    (9100, 10200),
    (14100, 15100),
    (16300, 17300),
    (26800, 27800),
    (29900, 29999),
]


# =========================
# Collect data
# =========================

reward_steps = []
reward_values = []

length_steps = []
length_values = []

for i, run in enumerate(runs):

    # -------- reward --------
    s, v = load_scalar(run, "Train/mean_reward")

    if i == 0:
        s = s[:-DROP_LAST * 4]
        v = v[:-DROP_LAST * 4]
    else:
        s = s[SKIP_FIRST:-DROP_LAST * 4]
        v = v[SKIP_FIRST:-DROP_LAST * 4]

    reward_steps.extend(s)
    reward_values.extend(v)

    # -------- episode length --------
    s, v = load_scalar(run, "Train/mean_episode_length")

    if i == 0:
        s = s[:-DROP_LAST]
        v = v[:-DROP_LAST]

        s = np.array(s)
        v = np.array(v)

        mask = v >= 850

        s = s[mask]
        v = v[mask]

    elif i == 4:
        s = s[SKIP_FIRST:-DROP_LAST]
        v = v[SKIP_FIRST:-DROP_LAST]
    else:
        s = s[SKIP_FIRST:]
        v = v[SKIP_FIRST:]

    length_steps.extend(s)
    length_values.extend(v)


# =========================
# Merge duplicates by step
# =========================

reward_dict = {}
for step, value in zip(reward_steps, reward_values):
    reward_dict[step] = value

reward_steps = sorted(reward_dict.keys())
reward_values = [reward_dict[s] for s in reward_steps]

print(f"Reward points: {len(reward_steps)}")

length_dict = {}
for step, value in zip(length_steps, length_values):
    length_dict[step] = value

length_steps = sorted(length_dict.keys())
length_values = [length_dict[s] for s in length_steps]

print(f"Episode points: {len(length_steps)}")


# =========================
# Utils
# =========================

def tensorboard_smooth(values, smoothing=0.6):
    values = np.asarray(values)

    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = smoothing * smoothed[i - 1] + (1 - smoothing) * values[i]

    return smoothed


def clip_spikes(x, y, max_jump=50):
    x = np.array(x)
    y = np.array(y)

    clean_y = y.copy()

    for i in range(1, len(y)):
        if abs(y[i] - y[i - 1]) > max_jump:
            clean_y[i] = clean_y[i - 1]

    return x, clean_y


# =========================
# Reward plot
# =========================

smooth = 0.90

reward_steps_c, reward_values_c = clip_spikes(
    reward_steps, reward_values, max_jump=50
)

reward_values_c = np.array(reward_values_c)
reward_values_c = np.clip(reward_values_c, -20, None)

reward_smooth = tensorboard_smooth(reward_values_c, smooth)

plt.figure(figsize=(14, 6))

plt.plot(
    reward_steps_c,
    reward_values_c,
    alpha=0.25,
    linewidth=1,
    label="Raw (cleaned)"
)

plt.plot(
    reward_steps_c,
    reward_smooth,
    linewidth=2,
    label=f"EMA smoothed (α={smooth})"
)

for i, (x1, x2) in enumerate(reward_zones):
    plt.axvspan(
        x1, x2,
        color="red",
        alpha=0.2,
        label="Removed data" if i == 0 else None
    )

plt.title("Mean Reward")
plt.xlabel("Training iteration")
plt.ylabel("Reward")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# Episode length plot
# =========================

smooth = 0.98
length_smooth = tensorboard_smooth(length_values, smooth)

plt.figure(figsize=(14, 6))

plt.plot(
    length_steps,
    length_values,
    alpha=0.25,
    linewidth=1,
    label="Raw data"
)

plt.plot(
    length_steps,
    length_smooth,
    linewidth=2,
    label=f"EMA smoothed (α={smooth})"
)

for i, (x1, x2) in enumerate(episode_zones):
    plt.axvspan(
        x1, x2,
        color="red",
        alpha=0.2,
        label="Removed data" if i == 0 else None
    )

plt.title("Mean Episode Length")
plt.xlabel("Training iteration")
plt.ylabel("Episode length")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
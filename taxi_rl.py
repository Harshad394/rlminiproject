import pickle
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np


Q_TABLE_PATH = Path("q_table.pkl")
ACTION_NAMES = {
    0: "South",
    1: "North",
    2: "East",
    3: "West",
    4: "Pickup",
    5: "Dropoff",
}
LANDMARKS = {
    "R": (0, 0),
    "G": (0, 4),
    "Y": (4, 0),
    "B": (4, 3),
}
PASSENGER_INDEX_TO_NAME = {
    0: "R",
    1: "G",
    2: "Y",
    3: "B",
    4: "IN_TAXI",
}
DESTINATION_INDEX_TO_NAME = {
    0: "R",
    1: "G",
    2: "Y",
    3: "B",
}


ProgressCallback = Callable[[int, int, float, float], None]


def train_taxi(
    episodes: int = 5000,
    max_steps: int = 200,
    learning_rate: float = 0.1,
    discount_rate: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.999,
    min_epsilon: float = 0.01,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[float]]:
    env = gym.make("Taxi-v3")
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))
    rewards: list[float] = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            best_next_action = int(np.argmax(q_table[next_state]))

            td_target = reward + discount_rate * q_table[next_state, best_next_action]
            td_error = td_target - q_table[state, action]
            q_table[state, action] += learning_rate * td_error

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if progress_callback and (episode + 1) % 100 == 0:
            avg_reward = float(np.mean(rewards[-100:]))
            progress_callback(episode + 1, episodes, epsilon, avg_reward)

    env.close()
    return q_table, rewards


def save_q_table(q_table: np.ndarray, path: Path = Q_TABLE_PATH) -> None:
    with path.open("wb") as file:
        pickle.dump(q_table, file)


def load_q_table(path: Path = Q_TABLE_PATH) -> np.ndarray:
    with path.open("rb") as file:
        return pickle.load(file)


def decode_state(state: int) -> dict[str, int | str]:
    destination_index = state % 4
    state //= 4
    passenger_index = state % 5
    state //= 5
    col = state % 5
    state //= 5
    row = state
    passenger_name = PASSENGER_INDEX_TO_NAME[passenger_index]
    destination_name = DESTINATION_INDEX_TO_NAME[destination_index]
    return {
        "row": row,
        "col": col,
        "passenger_index": passenger_index,
        "destination_index": destination_index,
        "passenger_name": passenger_name,
        "destination_name": destination_name,
    }


def run_demo_episode(
    q_table: np.ndarray,
    max_steps: int = 50,
    seed: int | None = None,
) -> list[dict[str, object]]:
    env = gym.make("Taxi-v3")
    state, _ = env.reset(seed=seed)
    total_reward = 0
    frames: list[dict[str, object]] = []

    for step in range(1, max_steps + 1):
        action = int(np.argmax(q_table[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        frames.append(
            {
                "step": step,
                "state": state,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "reward": reward,
                "total_reward": total_reward,
                "decoded_state": decode_state(state),
            }
        )

        state = next_state
        if terminated or truncated:
            frames.append(
                {
                    "step": step,
                    "state": state,
                    "action": None,
                    "action_name": "Finished",
                    "reward": 0,
                    "total_reward": total_reward,
                    "decoded_state": decode_state(state),
                    "done": True,
                }
            )
            break

    env.close()
    return frames

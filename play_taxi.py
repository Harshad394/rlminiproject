import time

from taxi_rl import ACTION_NAMES, Q_TABLE_PATH, load_q_table, run_demo_episode


def play_demo(max_steps: int = 50) -> None:
    if not Q_TABLE_PATH.exists():
        raise FileNotFoundError(
            "q_table.pkl not found. Run `python3 train_taxi.py` first."
        )

    q_table = load_q_table()
    frames = run_demo_episode(q_table, max_steps=max_steps)

    for frame in frames:
        if frame.get("done"):
            print("\nEpisode finished.")
            break

        decoded = frame["decoded_state"]
        print(f"\nStep {frame['step']}")
        print(
            "Taxi position: "
            f"({decoded['row']}, {decoded['col']}) | "
            f"Passenger: {decoded['passenger_name']} | "
            f"Destination: {decoded['destination_name']}"
        )
        print(
            f"Action: {frame['action']} ({ACTION_NAMES[frame['action']]}) | "
            f"Reward: {frame['reward']} | Total reward: {frame['total_reward']}"
        )
        time.sleep(0.3)


if __name__ == "__main__":
    play_demo()

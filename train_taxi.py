from taxi_rl import Q_TABLE_PATH, save_q_table, train_taxi


if __name__ == "__main__":
    def print_progress(episode: int, total: int, epsilon: float, avg_reward: float) -> None:
        print(
            f"Episode {episode}/{total} | "
            f"epsilon={epsilon:.4f} | average reward (last 100)={avg_reward:.2f}"
        )


    learned_q_table, _ = train_taxi(progress_callback=print_progress)
    save_q_table(learned_q_table)
    print(f"Saved trained Q-table to {Q_TABLE_PATH.resolve()}")

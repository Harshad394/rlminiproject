# Drive a Taxi with OpenAI Gym

This is a simple beginner project that trains an agent to solve the `Taxi-v3` environment using Q-learning.

Important note:
The old `openai-gym` package is no longer the recommended choice. This project uses `gymnasium`, which is the modern replacement and works almost the same way.

## Project files

- `train_taxi.py` trains a taxi agent with Q-learning
- `play_taxi.py` loads the trained Q-table and shows one demo episode
- `taxi_rl.py` contains the reusable RL logic
- `app.py` provides a browser UI
- `requirements.txt` contains the Python packages you need

## 1. Install Python tools

You already have `python3` on this machine.

Check it:

```bash
python3 --version
```

On Ubuntu/Debian, `ensurepip` is often disabled. Install the system packages first:

```bash
sudo apt update
sudo apt install python3.12-venv python3-pip
```

Then verify:

```bash
python3 -m pip --version
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install project dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 4. Train the taxi agent

```bash
python3 train_taxi.py
```

This will:

- create the `Taxi-v3` environment
- train a Q-table
- save the learned table to `q_table.pkl`
- print training progress and final reward

## 5. Watch the trained taxi play

```bash
python3 play_taxi.py
```

This runs one demo episode using the saved Q-table.

## 6. Run the UI

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

The UI lets you:

- train the taxi agent
- load the saved Q-table
- step through a demo visually
- see the taxi, passenger, destination, and rewards

## How the project works

The Taxi environment has:

- states: where the taxi is, where the passenger is, and the destination
- actions: move south, north, east, west, pick up, and drop off
- rewards: negative reward for wasting steps and positive reward for correct delivery

Q-learning updates a table using this rule:

```text
Q(state, action) = Q(state, action) + alpha * (reward + gamma * max(Q(next_state)) - Q(state, action))
```

In simple words:

- the agent tries actions
- it sees reward
- it updates the table
- over many episodes it learns better moves

## Implementation steps

If you want to build this yourself from scratch, follow this order:

1. Install `python3`, `pip`, and `venv`.
2. Create a virtual environment.
3. Install `gymnasium[toy-text]` and `numpy`.
4. Create the Taxi environment with `gym.make("Taxi-v3")`.
5. Read `env.observation_space.n` and `env.action_space.n`.
6. Create a Q-table with shape `(state_size, action_size)`.
7. Loop through many episodes.
8. Choose actions using epsilon-greedy exploration.
9. Update the Q-table after every step.
10. Decay epsilon slowly so the agent explores less over time.
11. Save the trained table with `pickle`.
12. Load the table and run a demo episode.

## Expected output

During training you should see logs like:

```text
Episode 500/5000 | epsilon=0.7783 | average reward (last 500)=-145.22
```

After training, the average reward should improve over time.

## Troubleshooting

If you get `No module named gymnasium`:

```bash
python3 -m pip install -r requirements.txt
```

If you get permission issues:

- make sure the virtual environment is active
- use `python3 -m pip ...` instead of plain `pip`

If you want next steps after this, the easiest upgrades are:

- plot reward curves
- compare random policy vs learned policy
- try FrozenLake after Taxi
- move from Q-learning to Deep Q-Networks later

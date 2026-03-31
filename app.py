import time

import streamlit as st

from taxi_rl import LANDMARKS, Q_TABLE_PATH, decode_state, load_q_table, run_demo_episode, save_q_table, train_taxi


GRID_SIZE = 5


def ensure_session_defaults() -> None:
    if "q_table" not in st.session_state:
        st.session_state.q_table = None
    if "frames" not in st.session_state:
        st.session_state.frames = []
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0
    if "training_summary" not in st.session_state:
        st.session_state.training_summary = ""
    if "auto_play" not in st.session_state:
        st.session_state.auto_play = False


def render_grid(decoded_state: dict[str, int | str]) -> str:
    taxi_row = int(decoded_state["row"])
    taxi_col = int(decoded_state["col"])
    passenger_name = str(decoded_state["passenger_name"])
    destination_name = str(decoded_state["destination_name"])
    passenger_in_taxi = passenger_name == "IN_TAXI"
    passenger_position = LANDMARKS.get(passenger_name)
    destination_position = LANDMARKS[destination_name]

    rows = []
    for row in range(GRID_SIZE):
        cells = []
        for col in range(GRID_SIZE):
            classes = ["cell"]
            labels = []

            for landmark_name, landmark_position in LANDMARKS.items():
                if landmark_position == (row, col):
                    labels.append(f"<span class='landmark'>{landmark_name}</span>")

            if destination_position == (row, col):
                classes.append("destination")
                labels.append("<span class='badge'>DEST</span>")

            if not passenger_in_taxi and passenger_position == (row, col):
                classes.append("passenger")
                labels.append("<span class='badge'>PASS</span>")

            if (taxi_row, taxi_col) == (row, col):
                classes.append("taxi")
                labels.append("<span class='taxi-label'>TAXI</span>")

            cell_html = f"<div class='{' '.join(classes)}'>{''.join(labels)}</div>"
            cells.append(cell_html)
        rows.append(f"<div class='row'>{''.join(cells)}</div>")

    in_taxi_text = "Yes" if passenger_in_taxi else "No"
    return f"""
    <style>
    .board {{
        display: inline-block;
        padding: 16px;
        border-radius: 20px;
        background: linear-gradient(180deg, #f6efe4 0%, #eadbc8 100%);
        border: 1px solid #d2baa0;
    }}
    .row {{
        display: flex;
    }}
    .cell {{
        width: 86px;
        height: 86px;
        margin: 4px;
        border-radius: 18px;
        background: #fffaf2;
        border: 1px solid #d9c8b2;
        position: relative;
        color: #3b2f2f;
        font-weight: 700;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }}
    .taxi {{
        background: #ffd166;
        border-color: #f2a900;
    }}
    .passenger {{
        box-shadow: inset 0 0 0 3px #2a9d8f;
    }}
    .destination {{
        box-shadow: inset 0 0 0 3px #e76f51;
    }}
    .landmark {{
        position: absolute;
        top: 8px;
        left: 10px;
        font-size: 0.8rem;
        color: #6b5b4d;
    }}
    .badge {{
        position: absolute;
        bottom: 8px;
        left: 8px;
        right: 8px;
        font-size: 0.72rem;
        color: #234;
        background: rgba(255,255,255,0.78);
        border-radius: 999px;
        padding: 2px 4px;
    }}
    .taxi-label {{
        font-size: 0.92rem;
        letter-spacing: 0.03em;
    }}
    .legend {{
        margin-top: 14px;
        font-size: 0.92rem;
        color: #4f4338;
    }}
    </style>
    <div class="board">
        {''.join(rows)}
        <div class="legend">
            Passenger in taxi: <strong>{in_taxi_text}</strong>
        </div>
    </div>
    """


def current_frame() -> dict[str, object] | None:
    frames = st.session_state.frames
    if not frames:
        return None
    return frames[st.session_state.frame_index]


st.set_page_config(page_title="Drive a Taxi", page_icon="T", layout="wide")
ensure_session_defaults()

st.title("Drive a Taxi using OpenAI Gym")
st.write(
    "Train a Q-learning taxi agent, then replay one episode on a visual 5x5 grid."
)

with st.sidebar:
    st.subheader("Training")
    episodes = st.slider("Episodes", min_value=500, max_value=10000, value=5000, step=500)
    max_steps = st.slider("Max steps per episode", min_value=50, max_value=300, value=200, step=25)
    learning_rate = st.slider("Learning rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    discount_rate = st.slider("Discount rate", min_value=0.50, max_value=0.999, value=0.99, step=0.01)
    epsilon_decay = st.slider("Epsilon decay", min_value=0.900, max_value=0.9999, value=0.999, step=0.0001)
    demo_seed = st.number_input("Demo seed", min_value=0, value=7, step=1)

    train_clicked = st.button("Train Agent", use_container_width=True)
    load_clicked = st.button("Load Saved Agent", use_container_width=True)
    demo_clicked = st.button("Generate Demo", use_container_width=True)

status_col, info_col = st.columns([1.1, 0.9])

with info_col:
    st.subheader("How it works")
    st.write(
        "The taxi must pick up the passenger at the correct landmark and deliver them to the target landmark while minimizing wasted moves."
    )
    st.write(
        "Rewards: `-1` per step, strong penalties for bad pickup/dropoff, and `+20` for a correct final dropoff."
    )
    if st.session_state.training_summary:
        st.success(st.session_state.training_summary)

if load_clicked:
    if Q_TABLE_PATH.exists():
        st.session_state.q_table = load_q_table()
        st.session_state.training_summary = f"Loaded trained agent from {Q_TABLE_PATH}."
    else:
        st.session_state.training_summary = "No saved Q-table found yet. Train the agent first."

if train_clicked:
    progress_bar = st.sidebar.progress(0, text="Training in progress...")
    progress_text = st.sidebar.empty()

    def update_progress(episode: int, total: int, epsilon: float, avg_reward: float) -> None:
        progress_bar.progress(
            min(int(episode / total * 100), 100),
            text=f"Training episode {episode}/{total}",
        )
        progress_text.write(
            f"Epsilon: {epsilon:.4f} | Average reward (last 100): {avg_reward:.2f}"
        )

    q_table, rewards = train_taxi(
        episodes=episodes,
        max_steps=max_steps,
        learning_rate=learning_rate,
        discount_rate=discount_rate,
        epsilon_decay=epsilon_decay,
        progress_callback=update_progress,
    )
    save_q_table(q_table)
    st.session_state.q_table = q_table
    st.session_state.training_summary = (
        f"Training complete. Final average reward over the last 100 episodes: "
        f"{sum(rewards[-100:]) / min(len(rewards), 100):.2f}"
    )
    progress_bar.progress(100, text="Training complete")

if demo_clicked and st.session_state.q_table is None and Q_TABLE_PATH.exists():
    st.session_state.q_table = load_q_table()

if demo_clicked:
    if st.session_state.q_table is None:
        st.session_state.training_summary = "Train or load an agent before generating a demo."
    else:
        st.session_state.frames = run_demo_episode(
            st.session_state.q_table,
            seed=int(demo_seed),
        )
        st.session_state.frame_index = 0

with status_col:
    st.subheader("Taxi Grid")
    frame = current_frame()
    if frame is None:
        placeholder_state = decode_state(123)
        st.markdown(render_grid(placeholder_state), unsafe_allow_html=True)
        st.info("Generate a demo to watch the taxi move.")
    else:
        st.markdown(
            render_grid(frame["decoded_state"]),
            unsafe_allow_html=True,
        )

        st.write(
            f"Step: {frame['step']} | Action: {frame['action_name']} | "
            f"Reward: {frame['reward']} | Total reward: {frame['total_reward']}"
        )

        nav1, nav2, nav3 = st.columns(3)
        with nav1:
            if st.button("Previous", use_container_width=True):
                st.session_state.auto_play = False
                st.session_state.frame_index = max(0, st.session_state.frame_index - 1)
                st.rerun()
        with nav2:
            if st.button("Next", use_container_width=True):
                st.session_state.auto_play = False
                st.session_state.frame_index = min(
                    len(st.session_state.frames) - 1,
                    st.session_state.frame_index + 1,
                )
                st.rerun()
        with nav3:
            if st.button("Auto Play", use_container_width=True):
                st.session_state.auto_play = True
                st.rerun()

if st.session_state.auto_play and st.session_state.frames:
    if st.session_state.frame_index < len(st.session_state.frames) - 1:
        time.sleep(0.35)
        st.session_state.frame_index += 1
        st.rerun()
    else:
        st.session_state.auto_play = False

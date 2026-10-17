import io
import math
import os
import shutil

import chess
import chess.engine
import chess.pgn
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chess PGN Analyzer",
    page_icon="♟️",
    layout="wide",
)

st.title("♟️ Automated Chess PGN Engine Analyzer")
st.write(
    "Enter a PGN game below to analyse moves using the Stockfish "
    "chess engine."
)


# ============================================================
# USER-ADJUSTABLE ANALYSIS SETTINGS
# ============================================================

DEFAULT_ENGINE_DEPTH = 12
DEFAULT_ANALYSED_MOVES = 20

MATE_SCORE = 100000

BLUNDER_THRESHOLD_CP = 300

# Controls how quickly candidate quality falls as centipawn loss increases.
# A larger value makes inferior candidate moves retain higher quality scores.
CANDIDATE_QUALITY_SCALE_CP = 200

# Candidate moves at or above this quality are counted as good alternatives.
GOOD_MOVE_QUALITY_THRESHOLD = 70.0


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "candidate_results" not in st.session_state:
    st.session_state.candidate_results = None

if "analysis_metadata" not in st.session_state:
    st.session_state.analysis_metadata = None


# ============================================================
# STOCKFISH AND SCORE HELPERS
# ============================================================

def find_stockfish():
    possible_paths = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        shutil.which("stockfish"),
    ]

    for path in possible_paths:
        if path and os.path.exists(path) and os.access(path, os.X_OK):
            return path

    return None


def score_to_cp(score, colour):
    if score is None:
        return 0

    numerical_score = score.pov(colour).score(
        mate_score=MATE_SCORE
    )

    if numerical_score is None:
        return 0

    return int(numerical_score)


def score_description(score, colour):
    if score is None:
        return "Unknown"

    pov_score = score.pov(colour)

    if pov_score.is_mate():
        mate_distance = pov_score.mate()

        if mate_distance is None:
            return "Mate"

        if mate_distance > 0:
            return f"Mate in {mate_distance}"

        return f"Mated in {abs(mate_distance)}"

    centipawn_score = pov_score.score()

    if centipawn_score is None:
        return "0.00"

    return f"{centipawn_score / 100:+.2f}"


def calculate_accuracy_score(engine_rank, analysed_moves):
    if engine_rank is None:
        return 0

    if engine_rank < 1 or engine_rank > analysed_moves:
        return 0

    return analysed_moves + 1 - engine_rank


def calculate_candidate_quality(best_score_cp, candidate_score_cp):
    evaluation_loss_cp = max(
        0,
        best_score_cp - candidate_score_cp,
    )

    quality = 100.0 * math.exp(
        -evaluation_loss_cp / CANDIDATE_QUALITY_SCALE_CP
    )

    return max(0.0, min(100.0, quality))


def candidate_quality_colour(quality):
    if quality >= 85:
        return "#1B5E20"

    if quality >= 70:
        return "#4CAF50"

    if quality >= 50:
        return "#FBC02D"

    if quality >= 25:
        return "#F57C00"

    return "#C62828"


def scope_includes_colour(analysis_scope, mover_colour):
    if analysis_scope == "Both players":
        return True

    if analysis_scope == "White only":
        return mover_colour == chess.WHITE

    if analysis_scope == "Black only":
        return mover_colour == chess.BLACK

    return False


def get_scope_description(
    analysis_scope,
    white_player,
    black_player,
):
    if analysis_scope == "White only":
        return f"White only: {white_player}"

    if analysis_scope == "Black only":
        return f"Black only: {black_player}"

    return (
        f"Both players: {white_player} and {black_player}"
    )


# ============================================================
# MOVE ANALYSIS
# ============================================================

def analyse_single_move(
    engine,
    board,
    played_move,
    depth,
    analysed_moves,
):
    mover_colour = board.turn
    legal_move_count = board.legal_moves.count()

    if legal_move_count < 1:
        raise ValueError(
            "The position has no legal moves to analyse."
        )

    requested_lines = min(
        analysed_moves,
        legal_move_count,
    )

    engine_result = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=requested_lines,
    )

    if isinstance(engine_result, dict):
        engine_lines = [engine_result]
    else:
        engine_lines = list(engine_result)

    ranked_moves = []
    played_move_line = None

    for returned_rank, info in enumerate(
        engine_lines,
        start=1,
    ):
        principal_variation = info.get("pv", [])
        returned_score = info.get("score")

        if not principal_variation:
            continue

        if returned_score is None:
            continue

        suggested_move = principal_variation[0]

        try:
            suggested_move_san = board.san(suggested_move)
        except (AssertionError, ValueError):
            suggested_move_san = suggested_move.uci()

        line_data = {
            "rank": returned_rank,
            "move": suggested_move,
            "move_uci": suggested_move.uci(),
            "score_cp": score_to_cp(
                returned_score,
                mover_colour,
            ),
            "evaluation": score_description(
                returned_score,
                mover_colour,
            ),
            "san": suggested_move_san,
        }

        ranked_moves.append(line_data)

        if suggested_move == played_move:
            played_move_line = line_data

    if not ranked_moves:
        raise RuntimeError(
            "Stockfish returned no usable candidate moves."
        )

    best_line = ranked_moves[0]
    best_score_cp = best_line["score_cp"]

    if played_move_line is None:
        played_analysis = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            root_moves=[played_move],
        )

        played_score_object = played_analysis.get("score")

        played_score_cp = score_to_cp(
            played_score_object,
            mover_colour,
        )

        played_evaluation = score_description(
            played_score_object,
            mover_colour,
        )

        engine_rank = None
    else:
        played_score_cp = played_move_line["score_cp"]
        played_evaluation = played_move_line["evaluation"]
        engine_rank = played_move_line["rank"]

    evaluation_loss_cp = max(
        0,
        best_score_cp - played_score_cp,
    )

    accuracy_score = calculate_accuracy_score(
        engine_rank,
        analysed_moves,
    )

    is_blunder = (
        evaluation_loss_cp >= BLUNDER_THRESHOLD_CP
    )

    candidate_rows = []

    for candidate in ranked_moves:
        candidate_loss_cp = max(
            0,
            best_score_cp - candidate["score_cp"],
        )

        candidate_quality = calculate_candidate_quality(
            best_score_cp,
            candidate["score_cp"],
        )

        candidate_rows.append({
            "candidate_rank": candidate["rank"],
            "candidate_move": candidate["san"],
            "candidate_move_uci": candidate["move_uci"],
            "candidate_evaluation": candidate["evaluation"],
            "candidate_score_cp": candidate["score_cp"],
            "candidate_loss_cp": candidate_loss_cp,
            "candidate_loss_pawns": candidate_loss_cp / 100,
            "candidate_quality": candidate_quality,
            "is_played_move": (
                candidate["move"] == played_move
            ),
        })

    # If the played move was outside the returned MultiPV lines,
    # append it so it can still appear on the candidate chart.
    if played_move_line is None:
        try:
            played_move_san = board.san(played_move)
        except (AssertionError, ValueError):
            played_move_san = played_move.uci()

        played_quality = calculate_candidate_quality(
            best_score_cp,
            played_score_cp,
        )

        candidate_rows.append({
            "candidate_rank": None,
            "candidate_move": played_move_san,
            "candidate_move_uci": played_move.uci(),
            "candidate_evaluation": played_evaluation,
            "candidate_score_cp": played_score_cp,
            "candidate_loss_cp": evaluation_loss_cp,
            "candidate_loss_pawns": (
                evaluation_loss_cp / 100
            ),
            "candidate_quality": played_quality,
            "is_played_move": True,
        })

    good_candidate_count = sum(
        1
        for candidate in candidate_rows
        if candidate["candidate_quality"]
        >= GOOD_MOVE_QUALITY_THRESHOLD
    )

    acceptable_candidate_count = sum(
        1
        for candidate in candidate_rows
        if candidate["candidate_quality"] >= 50
    )

    return {
        "accuracy_score": accuracy_score,
        "engine_rank": engine_rank,
        "is_in_analysed_moves": engine_rank is not None,
        "is_blunder": is_blunder,
        "evaluation_loss_cp": evaluation_loss_cp,
        "best_move_san": best_line["san"],
        "best_evaluation": best_line["evaluation"],
        "played_evaluation": played_evaluation,
        "best_score_cp": best_score_cp,
        "played_score_cp": played_score_cp,
        "candidate_rows": candidate_rows,
        "good_candidate_count": good_candidate_count,
        "acceptable_candidate_count": (
            acceptable_candidate_count
        ),
    }


# ============================================================
# RESULT DISPLAY HELPERS
# ============================================================

def safe_mean(dataframe, colour):
    colour_rows = dataframe[
        dataframe["colour"] == colour
    ]

    if colour_rows.empty:
        return None

    return colour_rows["accuracy_score"].mean()


def safe_blunder_count(dataframe, colour):
    colour_rows = dataframe[
        dataframe["colour"] == colour
    ]

    if colour_rows.empty:
        return None

    return int(colour_rows["is_blunder"].sum())


def display_metric_cards(
    results_df,
    analysis_scope,
    white_player,
    black_player,
    analysed_moves,
):
    metric_definitions = []

    if analysis_scope in ("Both players", "White only"):
        white_blunders = safe_blunder_count(
            results_df,
            "White",
        )

        white_average = safe_mean(
            results_df,
            "White",
        )

        metric_definitions.extend([
            (
                f"{white_player} Blunders",
                white_blunders,
            ),
            (
                f"{white_player} Avg Score",
                (
                    f"{white_average:.2f} / "
                    f"{analysed_moves}"
                ),
            ),
        ])

    if analysis_scope in ("Both players", "Black only"):
        black_blunders = safe_blunder_count(
            results_df,
            "Black",
        )

        black_average = safe_mean(
            results_df,
            "Black",
        )

        metric_definitions.extend([
            (
                f"{black_player} Blunders",
                black_blunders,
            ),
            (
                f"{black_player} Avg Score",
                (
                    f"{black_average:.2f} / "
                    f"{analysed_moves}"
                ),
            ),
        ])

    if not metric_definitions:
        return

    metric_columns = st.columns(
        len(metric_definitions)
    )

    for column, metric_definition in zip(
        metric_columns,
        metric_definitions,
    ):
        label, value = metric_definition
        column.metric(label=label, value=value)


def display_accuracy_chart(
    results_df,
    metadata,
):
    st.subheader("Move Accuracy Plot")

    analysed_moves = metadata["analysed_moves"]
    analysis_scope = metadata["analysis_scope"]
    white_player = metadata["white_player"]
    black_player = metadata["black_player"]
    event_name = metadata["event_name"]

    fig, ax = plt.subplots(figsize=(12, 5))

    white_moves = results_df[
        results_df["colour"] == "White"
    ]

    black_moves = results_df[
        results_df["colour"] == "Black"
    ]

    if not white_moves.empty:
        ax.plot(
            white_moves["move_number"],
            white_moves["accuracy_score"],
            marker="o",
            linewidth=1.5,
            label=f"White: {white_player}",
            color="#1f77b4",
        )

    if not black_moves.empty:
        ax.plot(
            black_moves["move_number"],
            black_moves["accuracy_score"],
            marker="s",
            linewidth=1.5,
            label=f"Black: {black_player}",
            color="#ff7f0e",
        )

    blunders = results_df[
        results_df["is_blunder"]
    ]

    if not blunders.empty:
        ax.scatter(
            blunders["move_number"],
            blunders["accuracy_score"],
            color="red",
            marker="x",
            s=150,
            linewidths=3,
            label="Blunder",
            zorder=10,
        )

        for _, row in blunders.iterrows():
            if row["colour"] == "White":
                move_prefix = (
                    f"{row['move_number']}."
                )
            else:
                move_prefix = (
                    f"{row['move_number']}..."
                )

            annotation = (
                f"{move_prefix}{row['Played Move']}\n"
                f"Loss: "
                f"{row['evaluation_loss_pawns'\]:.2f}"
            )

            ax.annotate(
                annotation,
                (
                    row["move_number"],
                    row["accuracy_score"],
                ),
                xytext=(5, 10),
                textcoords="offset points",
                fontsize=8,
                color="red",
            )

    scope_description = get_scope_description(
        analysis_scope,
        white_player,
        black_player,
    )

    ax.set_title(
        f"{white_player} vs {black_player} - "
        f"{event_name}\n{scope_description}"
    )

    ax.set_xlabel("Game move number")
    ax.set_ylabel("Move score")

    maximum_score = max(1, analysed_moves)

    ax.set_ylim(
        -0.35,
        maximum_score + 0.35,
    )

    tick_interval = max(
        1,
        math.ceil(maximum_score / 10),
    )

    ax.set_yticks(
        range(
            0,
            maximum_score + 1,
            tick_interval,
        )
    )

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def display_position_difficulty_chart(results_df):
    st.subheader("Good Moves Available by Position")

    st.caption(
        "This chart shows how many candidate moves scored at "
        f"least {GOOD_MOVE_QUALITY_THRESHOLD:.0f}% relative "
        "to the engine's best move. Taller bars indicate a "
        "more forgiving position with several strong options."
    )

    if results_df.empty:
        st.info("No analysed positions are available.")
        return

    plot_df = results_df.copy()

    bar_colours = [
        "#1f77b4"
        if colour == "White"
        else "#ff7f0e"
        for colour in plot_df["colour"]
    ]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.bar(
        range(len(plot_df)),
        plot_df["good_candidate_count"],
        color=bar_colours,
        alpha=0.85,
    )

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(
        plot_df["Move"],
        rotation=70,
        ha="right",
        fontsize=8,
    )

    ax.set_xlabel("Played move")
    ax.set_ylabel("Good candidate moves")
    ax.set_ylim(
        0,
        max(
            1,
            int(
                plot_df[
                    "good_candidate_count"
                ].max()
            ) + 1,
        ),
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def display_candidate_chart(
    results_df,
    candidates_df,
):
    st.subheader("Candidate Moves for a Selected Position")

    if results_df.empty or candidates_df.empty:
        st.info(
            "No candidate-move information is available."
        )
        return

    position_options = results_df[
        "position_label"
    ].tolist()

    selected_position = st.selectbox(
        "Select an analysed move:",
        options=position_options,
        key="selected_candidate_position",
    )

    selected_result = results_df[
        results_df["position_label"]
        == selected_position
    ].iloc[0]

    selected_candidates = candidates_df[
        candidates_df["position_id"]
        == selected_result["position_id"]
    ].copy()

    if selected_candidates.empty:
        st.info(
            "No candidate moves were stored for this position."
        )
        return

    selected_candidates = selected_candidates.sort_values(
        by=[
            "candidate_quality",
            "candidate_score_cp",
        ],
        ascending=[True, True],
    ).reset_index(drop=True)

    chart_labels = []

    for _, candidate in selected_candidates.iterrows():
        rank_value = candidate["candidate_rank"]

        if pd.isna(rank_value):
            rank_text = "outside analysed lines"
        else:
            rank_text = f"rank {int(rank_value)}"

        played_text = (
            "  ← PLAYED"
            if candidate["is_played_move"]
            else ""
        )

        chart_labels.append(
            f"{candidate['candidate_move']} "
            f"({rank_text}){played_text}"
        )

    bar_colours = [
        candidate_quality_colour(quality)
        for quality in selected_candidates[
            "candidate_quality"
        ]
    ]

    fig_height = max(
        4.5,
        len(selected_candidates) * 0.42,
   )

    fig, ax = plt.subplots(
        figsize=(12, fig_height)
    )

    bars = ax.barh(
        range(len(selected_candidates)),
        selected_candidates[
            "candidate_quality"
        ],
        color=bar_colours,
        alpha=0.9,
    )

    for bar, (_, candidate) in zip(
        bars,
        selected_candidates.iterrows(),
    ):
        if candidate["is_played_move"]:
            bar.set_edgecolor("#000000")
            bar.set_linewidth(3.0)
        else:
            bar.set_edgecolor("#424242")
            bar.set_linewidth(0.5)

        quality_text = (
            f"{candidate['candidate_quality']:.1f}%"
        )

        evaluation_text = (
            candidate["candidate_evaluation"]
        )

        annotation_text = (
            f"{quality_text} | Eval "
            f"{evaluation_text}"
        )

        text_x = min(
            candidate["candidate_quality"] + 1.0,
            92.0,
        )

        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            annotation_text,
            va="center",
            fontsize=8,
            color="#111111",
        )

    ax.set_yticks(
        range(len(selected_candidates))
    )

    ax.set_yticklabels(
        chart_labels,
        fontsize=9,
    )

    ax.set_xlim(0, 112)
    ax.set_xlabel(
        "Candidate quality relative to the best move (%)"
    )

    ax.set_title(
        f"{selected_result['position_label']}\n"
        f"Best move: "
        f"{selected_result['Best Move']} | "
        f"Played move: "
        f"{selected_result['Played Move']}"
    )

    ax.axvline(
        GOOD_MOVE_QUALITY_THRESHOLD,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label=(
            f"Good-move threshold "
            f"({GOOD_MOVE_QUALITY_THRESHOLD:.0f}%)"
        ),
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.25,
    )

    ax.legend(loc="lower right")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    detail_columns = [
        "candidate_rank",
        "candidate_move",
        "candidate_evaluation",
        "candidate_loss_pawns",
        "candidate_quality",
        "is_played_move",
    ]

    candidate_detail_df = selected_candidates[
        detail_columns
    ].copy()

    candidate_detail_df = candidate_detail_df.rename(
        columns={
            "candidate_rank": "Rank",
            "candidate_move": "Candidate Move",
            "candidate_evaluation": "Evaluation",
            "candidate_loss_pawns": (
                "Loss Compared with Best (pawns)"
            ),
            "candidate_quality": "Quality (%)",
            "is_played_move": "Played Move",
        }
    )

    candidate_detail_df[
        "Loss Compared with Best (pawns)"
    ] = candidate_detail_df[
        "Loss Compared with Best (pawns)"
    ].round(2)

    candidate_detail_df[
        "Quality (%)"
    ] = candidate_detail_df[
        "Quality (%)"
    ].round(1)

    st.dataframe(
        candidate_detail_df.sort_values(
            by="Quality (%)",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )


def display_detailed_log(results_df):
    st.subheader("Detailed Move-by-Move Log")

    display_columns = [
        "Move",
        "Player",
        "Played Move",
        "Best Move",
        "Played Evaluation",
        "Best Evaluation",
        "Engine Rank",
        "Good Moves Available",
        "Evaluation Loss",
        "Blunder",
    ]

    log_df = results_df[
        display_columns
    ].copy()

    st.dataframe(
        log_df,
        use_container_width=True,
        hide_index=True,
    )


def display_saved_results():
    results_df = st.session_state.analysis_results
    candidates_df = st.session_state.candidate_results
    metadata = st.session_state.analysis_metadata

    if results_df is None:
        return

    if candidates_df is None:
        return

    if metadata is None:
        return

    if results_df.empty:
        st.warning(
            "The selected analysis scope contained no moves."
        )
        return

    st.success("Analysis complete.")

    st.info(
        "Analysed game: "
        f"**{metadata['white_player']}** vs "
        f"**{metadata['black_player']}**  \n"
        f"Scope: **"
        f"{get_scope_description(
            metadata['analysis_scope'],
            metadata['white_player'],
            metadata['black_player'],
        )}"
        f"**"
    )

    display_metric_cards(
        results_df,
        metadata["analysis_scope"],
        metadata["white_player"],
        metadata["black_player"],
        metadata["analysed_moves"],
    )

    display_accuracy_chart(
        results_df,
        metadata,
    )

    display_position_difficulty_chart(
        results_df,
    )

    display_candidate_chart(
        results_df,
        candidates_df,
    )

    display_detailed_log(results_df)


# ============================================================
# USER INPUT
# ============================================================

pgn_input = st.text_area(
    "Paste PGN here:",
    height=250,
    placeholder=(
        '[Event "Live Chess"]\n'
        '[White "White Player"]\n'
        '[Black "Black Player"]\n\n'
        "1. e4 c5 2. Nf3 d6"
    ),
)

analysis_scope = st.radio(
    "Which side should be analysed?",
    options=[
        "Both players",
        "White only",
        "Black only",
    ],
    horizontal=True,
)

settings_column_1, settings_column_2 = st.columns(2)

with settings_column_1:
    engine_depth = st.number_input(
        "Stockfish analysis depth",
        min_value=1,
        max_value=30,
        value=DEFAULT_ENGINE_DEPTH,
        step=1,
        help=(
            "Higher depth normally improves analysis but "
            "takes longer."
        ),
    )

with settings_column_2:
    analysed_moves = st.number_input(
        "Candidate moves per position",
        min_value=1,
        max_value=50,
        value=DEFAULT_ANALYSED_MOVES,
        step=1,
        help=(
            "The number of Stockfish candidate moves requested "
            "for each analysed position."
        ),
    )

run_analysis = st.button(
    "Run Stockfish Analysis",
    type="primary",
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if run_analysis:
    st.session_state.analysis_results = None
    st.session_state.candidate_results = None
    st.session_state.analysis_metadata = None

    cleaned_pgn = pgn_input.strip()

    if not cleaned_pgn:
        st.warning("Please enter a valid PGN first.")

    else:
        engine_path = find_stockfish()

        if not engine_path:
            st.error(
                "Stockfish was not found on the server. "
                "Install Stockfish and ensure that its "
                "executable is available at "
                "'/usr/games/stockfish', "
                "'/usr/bin/stockfish', or on PATH."
            )

        else:
            pgn_stream = io.StringIO(cleaned_pgn)
            game = chess.pgn.read_game(pgn_stream)

            if game is None:
                st.error(
                    "Failed to parse the PGN. "
                    "Check the PGN formatting."
                )

            elif game.errors:
                error_messages = "\n".join(
                    f"- {error}"
                    for error in game.errors
                )

                st.error(
                    "The PGN contained parsing errors:\n\n"
                    f"{error_messages}"
                )

            else:
                white_player = game.headers.get(
                    "White",
                    "White",
                )

                black_player = game.headers.get(
                    "Black",
                    "Black",
                )

                event_name = game.headers.get(
                    "Event",
                    "Chess Game",
                )

                board = game.board()
                moves_to_analyse = list(
                    game.mainline_moves()
                )

                total_ply_count = len(
                    moves_to_analyse
                )

                if total_ply_count == 0:
                    st.error(
                        "The PGN does not contain any moves."
                    )

                else:
                    selected_move_count = 0
                    count_board = game.board()

                    for move in moves_to_analyse:
                        if scope_includes_colour(
                            analysis_scope,
                            count_board.turn,
                        ):
                            selected_move_count += 1

                        count_board.push(move)

                    if selected_move_count == 0:
                        st.error(
                            "The selected side has no moves "
                            "in this PGN."
                        )

                    else:
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()

                        result_rows = []
                        candidate_rows = []

                        engine = None
                        analysis_failed = False

                        try:
                            engine = (
                                chess.engine.SimpleEngine
                                .popen_uci(engine_path)
                            )

                            analysed_position_index = 0

                            for ply_index, played_move in enumerate(
                                moves_to_analyse,
                                start=1,
                            ):
                                mover_colour = board.turn
                                include_move = (
                                    scope_includes_colour(
                                        analysis_scope,
                                        mover_colour,
                                    )
                                )

                                full_move_number = (
                                    board.fullmove_number
                                )

                                mover_name = (
                                    "White"
                                    if mover_colour
                                    == chess.WHITE
                                    else "Black"
                                )

                                player_name = (
                                    white_player
                                    if mover_colour
                                    == chess.WHITE
                                    else black_player
                                )

                                try:
                                    played_san = board.san(
                                        played_move
                                    )
                                except (
                                    AssertionError,
                                    ValueError,
                                ) as error:
                                    raise ValueError(
                                        "An illegal move was "
                                        "encountered at ply "
                                        f"{ply_index}: {error}"
                                    ) from error

                                if include_move:
                                    analysed_position_index += 1

                                    status_text.text(
                                        "Analysing selected "
                                        "position "
                                        f"{analysed_position_index}"
                                        f"/{selected_move_count}: "
                                        f"{full_move_number}"
                                        f"{'.' if mover_name == 'White' else '...'}"
                                        f"{played_san}"
                                    )

                                    analysis = (
                                        analyse_single_move(
                                            engine=engine,
                                            board=board,
                                            played_move=played_move,
                                            depth=int(
                                                engine_depth
                                            ),
                                            analysed_moves=int(
                                                analysed_moves
                                            ),
                                        )
                                    )

                                    if mover_name == "White":
                                        move_label = (
                                            f"{full_move_number}. "
                                            f"{played_san}"
                                        )
                                    else:
                                        move_label = (
                                            f"{full_move_number}... "
                                            f"{played_san}"
                                        )

                                    position_id = (
                                        f"{ply_index}_"
                                        f"{mover_name}_"
                                        f"{played_move.uci()}"
                                    )

                                    position_label = (
                                        f"{move_label} — "
                                        f"{player_name}"
                                    )

                                    result_rows.append({
                                        "position_id": position_id,
                                        "position_label": (
                                            position_label
                                        ),
                                        "game_number": 1,
                                        "white": white_player,
                                        "black": black_player,
                                        "event": event_name,
                                        "move_number": (
                                            full_move_number
                                        ),
                                        "ply_number": ply_index,
                                        "colour": mover_name,
                                        "Move": move_label,
                                        "Player": player_name,
                                        "Played Move": played_san,
                                        "Best Move": (
                                            analysis[
                                                "best_move_san"
                                            ]
                                        ),
                                        "Played Evaluation": (
                                            analysis[
                                                "played_evaluation"
                                            ]
                                        ),
                                        "Best Evaluation": (
                                            analysis[
                                                "best_evaluation"
                                            ]
                                        ),
                                        "Engine Rank": (
                                            analysis[
                                                "engine_rank"
                                            ]
                                            if analysis[
                                                "engine_rank"
                                            ]
                                            is not None
                                            else "Outside top "
                                            f"{int(analysed_moves)}"
                                        ),
                                        "accuracy_score": (
                                            analysis[
                                                "accuracy_score"
                                            ]
                                        ),
                                        "evaluation_loss_pawns": (
                                            analysis[
                                                "evaluation_loss_cp"
                                            ]
                                            / 100
                                        ),
                                        "Evaluation Loss": (
                                            f"{analysis[
                                                'evaluation_loss_cp'
                                            ] / 100:.2f} pawns"
                                        ),
                                        "Blunder": (
                                            "🚨 BLUNDER"
                                            if analysis[
                                                "is_blunder"
                                            ]
                                            else "OK"
                                        ),
                                        "is_blunder": (
                                            analysis[
                                                "is_blunder"
                                            ]
                                        ),
                                        "good_candidate_count": (
                                            analysis[
                                                "good_candidate_count"
                                            ]
                                        ),
                                        "Good Moves Available": (
                                            analysis[
                                                "good_candidate_count"
                                            ]
                                        ),
                                        "acceptable_candidate_count": (
                                            analysis[
                                                "acceptable_candidate_count"
                                            ]
                                        ),
                                    })

                                    for candidate in analysis[
                                        "candidate_rows"
                                    ]:
                                        candidate_row = (
                                            candidate.copy()
                                        )

                                        candidate_row.update({
                                            "position_id": (
                                                position_id
                                            ),
                                            "position_label": (
                                                position_label
                                            ),
                                            "move_number": (
                                                full_move_number
                                            ),
                                            "ply_number": (
                                                ply_index
                                            ),
                                            "colour": (
                                                mover_name
                                            ),
                                            "player": (
                                                player_name
                                            ),
                                            "played_move": (
                                                played_san
                                            ),
                                            "best_move": (
                                                analysis[
                                                    "best_move_san"
                                                ]
                                            ),
                                        })

                                        candidate_rows.append(
                                            candidate_row
                                        )

                                board.push(played_move)

                                progress_bar.progress(
                                    ply_index
                                    / total_ply_count
                                )

                        except (
                            chess.engine.EngineError,
                            chess.engine.EngineTerminatedError,
                            OSError,
                            ValueError,
                            RuntimeError,
                        ) as error:
                            analysis_failed = True

                            st.error(
                                "Analysis failed: "
                                f"{type(error).__name__}: "
                                f"{error}"
                            )

                        finally:
                            if engine is not None:
                                try:
                                    engine.quit()
                                except (
                                    chess.engine.EngineError,
                                    chess.engine
                                    .EngineTerminatedError,
                                    OSError,
                                ):
                                    pass

                        if not analysis_failed:
                            results_df = pd.DataFrame(
                                result_rows
                            )

                            candidates_df = pd.DataFrame(
                                candidate_rows
                            )

                            if results_df.empty:
                                st.error(
                                    "No moves were analysed."
                                )

                            else:
                                st.session_state[
                                    "analysis_results"
                                ] = results_df

                                st.session_state[
                                    "candidate_results"
                                ] = candidates_df

                                st.session_state[
                                    "analysis_metadata"
                                ] = {
                                    "white_player": (
                                        white_player
                                    ),
                                    "black_player": (
                                        black_player
                                    ),
                                    "event_name": (
                                        event_name
                                    ),
                                    "analysis_scope": (
                                        analysis_scope
                                    ),
                                    "engine_depth": int(
                                        engine_depth
                                    ),
                                    "analysed_moves": int(
                                        analysed_moves
                                    ),
                                    "engine_path": (
                                        engine_path
                                    ),
                                }

                                progress_bar.progress(1.0)
                                status_text.success(
                                    "Analysis complete."
                                )


# ============================================================
# DISPLAY CURRENT OR SAVED RESULTS
# ============================================================

display_saved_results()

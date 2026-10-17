pgn_text = """
"""

!apt-get update -qq
!apt-get install -y stockfish
!pip install -q python-chess pandas matplotlib tqdm

CONFIG = {
    # ========================================================
    # STOCKFISH SETTINGS
    # ========================================================

    "stockfish_path": "/usr/games/stockfish",

    # Engine search depth.
    "depth": 12,

    # Colab processing resources allocated to Stockfish.
    "threads": 2,
    "hash_size_mb": 256,

    # ========================================================
    # MOVE ANALYSIS AND SCORING
    # ========================================================

    # This controls both:
    # 1. The number of candidate moves requested from Stockfish.
    # 2. The maximum number of points available.
    #
    # For example, if this is 20:
    # Rank 1  = 20 points
    # Rank 2  = 19 points
    # Rank 3  = 18 points
    # ...
    # Rank 20 = 1 point
    # Outside the top 20 = 0 points
    "analysed_moves": 20,

    # A move is marked as a blunder when it loses at least this
    # many centipawns compared with the best engine move.
    #
    # 100 centipawns is approximately one pawn.
    "blunder_threshold_cp": 300,

    # ========================================================
    # PROGRESS REPORTING
    # ========================================================

    # Display a live progress bar.
    "show_progress_bar": True,

    # Show the move currently being analysed.
    "show_current_move": True,

    # Print a detailed progress message after this many plies.
    # One ply is one move by one player.
    "progress_update_every_plies": 5,

    # ========================================================
    # GRAPH SETTINGS
    # ========================================================

    "show_graphs": True,
    "graph_width": 15,
    "graph_height": 6,

    # Add text labels to blunders on the graphs.
    "annotate_blunders": True,

    # ========================================================
    # OUTPUT SETTINGS
    # ========================================================

    "round_decimal_places": 2,

    # Automatically create CSV files when analysis finishes.
    "automatically_save_csv": True,

    "detailed_csv_filename": "chess_move_analysis.csv",
    "summary_csv_filename": "chess_game_summary.csv",
}

import io
import os
import shutil
import time

import chess
import chess.engine
import chess.pgn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

MATE_SCORE = 100_000

def validate_configuration(config):
    """Validate all configuration settings."""

    positive_integer_settings = [
        "depth",
        "threads",
        "hash_size_mb",
        "analysed_moves",
        "blunder_threshold_cp",
        "graph_width",
        "graph_height",
        "progress_update_every_plies",
    ]

    for setting_name in positive_integer_settings:
        value = config.get(setting_name)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                f"CONFIG['{setting_name}'] must be "
                "a positive integer."
            )

    decimal_places = config.get(
        "round_decimal_places"
    )

    if (
        isinstance(decimal_places, bool)
        or not isinstance(decimal_places, int)
        or decimal_places < 0
    ):
        raise ValueError(
            "CONFIG['round_decimal_places'] must be "
            "a non-negative integer."
        )

    if config["analysed_moves"] > 500:
        raise ValueError(
            "CONFIG['analysed_moves'] must be 500 or fewer."
        )

    boolean_settings = [
        "show_progress_bar",
        "show_current_move",
        "show_graphs",
        "annotate_blunders",
        "automatically_save_csv",
    ]

    for setting_name in boolean_settings:
        if not isinstance(
            config.get(setting_name),
            bool
        ):
            raise ValueError(
                f"CONFIG['{setting_name}'] must be "
                "True or False."
            )

    string_settings = [
        "stockfish_path",
        "detailed_csv_filename",
        "summary_csv_filename",
    ]

    for setting_name in string_settings:
        value = config.get(setting_name)

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"CONFIG['{setting_name}'] must be "
                "a non-empty string."
            )

def find_stockfish(preferred_path):
    """Find the Stockfish executable."""

    possible_paths = [
        preferred_path,
        shutil.which("stockfish"),
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
    ]

    checked_paths = set()

    for path in possible_paths:
        if not path:
            continue

        if path in checked_paths:
            continue

        checked_paths.add(path)

        if (
            os.path.isfile(path)
            and os.access(path, os.X_OK)
        ):
            return path

    raise FileNotFoundError(
        "Stockfish could not be found. Run Cell 1 "
        "and check CONFIG['stockfish_path']."
    )

def load_games_from_pgn(pgn_string):
    """Load all games from a PGN string."""

    if not isinstance(pgn_string, str):
        raise TypeError(
            "The PGN input must be a string."
        )

    if not pgn_string.strip():
        raise ValueError(
            "The PGN string is empty."
        )

    games = []
    pgn_stream = io.StringIO(pgn_string)

    while True:
        game = chess.pgn.read_game(pgn_stream)

        if game is None:
            break

        if game.errors:
            error_text = "; ".join(
                str(error)
                for error in game.errors
            )

            print(
                f"PGN parser warning: {error_text}"
            )

        games.append(game)

    if not games:
        raise ValueError(
            "No games could be read from the PGN string."
        )

    return games

def count_total_plies(games):
    """Count all player moves across all games."""

    total_plies = 0

    for game in games:
        total_plies += sum(
            1
            for _ in game.mainline_moves()
        )

    return total_plies

def score_to_cp(score, colour):
    """
    Convert an engine score to centipawns from the
    moving player's perspective.
    """

    numerical_score = score.pov(colour).score(
        mate_score=MATE_SCORE
    )

    if numerical_score is None:
        raise ValueError(
            "Stockfish returned an unusable evaluation."
        )

    return int(numerical_score)

def score_description(score, colour):
    """
    Return a readable evaluation from the moving
    player's perspective.
    """

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
        return "Unknown"

    return f"{centipawn_score / 100:+.2f}"

def calculate_accuracy_score(
    engine_rank,
    analysed_moves
):
    """
    Convert engine rank to a score.

    For 20 analysed moves:
        Rank 1 = 20
        Rank 2 = 19
        Rank 20 = 1
        Outside top 20 = 0
    """

    if engine_rank is None:
        return 0

    if engine_rank < 1:
        return 0

    if engine_rank > analysed_moves:
        return 0

    return analysed_moves + 1 - engine_rank

def format_elapsed_time(seconds):
    """Convert seconds to HH:MM:SS."""

    total_seconds = max(
        0,
        int(round(seconds))
    )

    hours, remainder = divmod(
        total_seconds,
        3600
    )

    minutes, remaining_seconds = divmod(
        remainder,
        60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )

def build_move_label(
    move_number,
    colour_name,
    move_san
):
    """Build a conventional move label."""

    if colour_name == "White":
        return f"{move_number}.{move_san}"

    return f"{move_number}...{move_san}"

def analyse_single_move(
    engine,
    board,
    played_move,
    depth,
    analysed_moves,
    blunder_threshold_cp
):
    """Analyse one move from the current position."""

    if played_move not in board.legal_moves:
        raise ValueError(
            "Illegal move encountered: "
            f"{played_move.uci()}"
        )

    mover_colour = board.turn
    legal_move_count = board.legal_moves.count()

    if legal_move_count < 1:
        raise ValueError(
            "The position contains no legal moves."
        )

    requested_lines = min(
        analysed_moves,
        legal_move_count
    )

    engine_result = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=requested_lines,
    )

    if isinstance(engine_result, dict):
        engine_lines = [engine_result]
    else:
        engine_lines = engine_result

    ranked_moves = []
    played_move_line = None

    for returned_rank, info in enumerate(
        engine_lines,
        start=1
    ):
        principal_variation = info.get(
            "pv",
            []
        )

        returned_score = info.get("score")

        if not principal_variation:
            continue

        if returned_score is None:
            continue

        suggested_move = principal_variation[0]

        line_data = {
            "rank": returned_rank,
            "move": suggested_move,
            "uci": suggested_move.uci(),
            "san": board.san(suggested_move),
            "score_cp": score_to_cp(
                returned_score,
                mover_colour
            ),
            "evaluation": score_description(
                returned_score,
                mover_colour
            ),
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

        played_score_object = played_analysis.get(
            "score"
        )

        if played_score_object is None:
            raise RuntimeError(
                "Stockfish returned no score for the "
                "played move."
            )

        played_score_cp = score_to_cp(
            played_score_object,
            mover_colour
        )

        played_evaluation = score_description(
            played_score_object,
            mover_colour
        )

        engine_rank = None

    else:
        played_score_cp = played_move_line[
            "score_cp"
        ]

        played_evaluation = played_move_line[
            "evaluation"
        ]

        engine_rank = played_move_line["rank"]

    accuracy_score = calculate_accuracy_score(
        engine_rank,
        analysed_moves
    )

    evaluation_loss_cp = max(
        0,
        best_score_cp - played_score_cp
    )

    is_blunder = (
        evaluation_loss_cp
        >= blunder_threshold_cp
    )

    return {
        "accuracy_score": accuracy_score,
        "engine_rank": engine_rank,
        "is_in_analysed_moves": (
            engine_rank is not None
        ),
        "is_blunder": is_blunder,
        "evaluation_loss_cp": evaluation_loss_cp,
        "best_score_cp": best_score_cp,
        "played_score_cp": played_score_cp,
        "best_evaluation": best_line[
            "evaluation"
        ],
        "played_evaluation": played_evaluation,
        "best_move_san": best_line["san"],
        "best_move_uci": best_line["uci"],
        "ranked_moves": ranked_moves,
    }

def add_candidate_columns(
    row,
    ranked_moves,
    analysed_moves
):
    """
    Add all candidate moves and evaluations to a row.
    """

    column_width = max(
        2,
        len(str(analysed_moves))
    )

    for rank_number in range(
        1,
        analysed_moves + 1
    ):
        column_number = str(
            rank_number
        ).zfill(column_width)

        list_index = rank_number - 1

        if list_index < len(ranked_moves):
            line = ranked_moves[list_index]

            row[
                f"top_{column_number}_san"
            ] = line["san"]

            row[
                f"top_{column_number}_uci"
            ] = line["uci"]

            row[
                f"top_{column_number}_evaluation"
            ] = line["evaluation"]

            row[
                f"top_{column_number}_score_cp"
            ] = line["score_cp"]

        else:
            row[
                f"top_{column_number}_san"
            ] = None

            row[
                f"top_{column_number}_uci"
            ] = None

            row[
                f"top_{column_number}_evaluation"
            ] = None

            row[
                f"top_{column_number}_score_cp"
            ] = None

def create_game_summary(
    move_results,
    analysed_moves
):
    """Create a summary for each player and game."""

    summary = (
        move_results
        .groupby(
            [
                "game_number",
                "event",
                "white",
                "black",
                "result",
                "player",
                "colour",
            ],
            as_index=False
        )
        .agg(
            moves=(
                "played_move_san",
                "count"
            ),
            total_score=(
                "accuracy_score",
                "sum"
            ),
            average_score=(
                "accuracy_score",
                "mean"
            ),
            average_accuracy_percent=(
                "accuracy_percent",
                "mean"
            ),
            moves_in_analysed_range=(
                "is_in_analysed_moves",
                "sum"
            ),
            blunders=(
                "is_blunder",
                "sum"
            ),
            average_loss_cp=(
                "evaluation_loss_cp",
                "mean"
            ),
            maximum_loss_cp=(
                "evaluation_loss_cp",
                "max"
            ),
        )
    )

    summary[
        "analysed_moves_setting"
    ] = analysed_moves

    summary[
        "maximum_possible_score"
    ] = summary["moves"] * analysed_moves

    summary[
        "analysed_range_percent"
    ] = (
        summary["moves_in_analysed_range"]
        / summary["moves"]
        * 100
    )

    columns_to_round = [
        "average_score",
        "average_accuracy_percent",
        "analysed_range_percent",
        "average_loss_cp",
        "maximum_loss_cp",
    ]

    summary[columns_to_round] = (
        summary[columns_to_round].round(2)
    )

    return summary

def plot_game_accuracy(
    move_results,
    game_number,
    config
):
    """Plot move scores for one game."""

    selected_game = move_results.loc[
        move_results["game_number"]
        == game_number
    ].copy()

    if selected_game.empty:
        raise ValueError(
            f"No results found for game {game_number}."
        )

    white_name = selected_game[
        "white"
    ].iloc[0]

    black_name = selected_game[
        "black"
    ].iloc[0]

    event_name = selected_game[
        "event"
    ].iloc[0]

    maximum_score = config[
        "analysed_moves"
    ]

    white_moves = selected_game.loc[
        selected_game["colour"] == "White"
    ]

    black_moves = selected_game.loc[
        selected_game["colour"] == "Black"
    ]

    plt.figure(
        figsize=(
            config["graph_width"],
            config["graph_height"],
        )
    )

    if not white_moves.empty:
        plt.plot(
            white_moves["move_number"],
            white_moves["accuracy_score"],
            marker="o",
            linewidth=1.5,
            label=f"White: {white_name}",
        )

    if not black_moves.empty:
        plt.plot(
            black_moves["move_number"],
            black_moves["accuracy_score"],
            marker="s",
            linewidth=1.5,
            label=f"Black: {black_name}",
        )

    blunders = selected_game.loc[
        selected_game["is_blunder"]
    ]

    if not blunders.empty:
        plt.scatter(
            blunders["move_number"],
            blunders["accuracy_score"],
            color="red",
            marker="x",
            s=150,
            linewidths=3,
            label="Blunder",
            zorder=10,
        )

        if config["annotate_blunders"]:
            for _, row in blunders.iterrows():
                if row["colour"] == "White":
                    move_prefix = (
                        f'{row["move_number"]}.'
                    )
                else:
                    move_prefix = (
                        f'{row["move_number"]}...'
                    )

                annotation = (
                    f'{move_prefix}'
                    f'{row["played_move_san"]}\n'
                    f'Loss: '
                    f'{row["evaluation_loss_pawns"]:.2f}'
                )

                plt.annotate(
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

    plt.title(
        f"Game {game_number}: "
        f"{white_name} vs {black_name}\n"
        f"{event_name}"
    )

    plt.xlabel("Game move number")
    plt.ylabel("Move score")

    if maximum_score <= 30:
        plt.yticks(
            range(
                0,
                maximum_score + 1
            )
        )

    else:
        tick_interval = max(
            1,
            maximum_score // 10
        )

        ticks = list(
            range(
                0,
                maximum_score + 1,
                tick_interval
            )
        )

        if ticks[-1] != maximum_score:
            ticks.append(maximum_score)

        plt.yticks(ticks)

    plt.ylim(
        -0.35,
        maximum_score + 0.35
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_games(
    move_results,
    config
):
    """Plot every analysed game."""

    game_numbers = sorted(
        move_results[
            "game_number"
        ].unique()
    )

    for game_number in game_numbers:
        plot_game_accuracy(
            move_results,
            int(game_number),
            config,
        )


def analyse_pgn(
    pgn_string,
    config=None
):
    """
    Analyse all main-line moves in all supplied games.
    """

    active_config = CONFIG.copy()

    if config is not None:
        active_config.update(config)

    validate_configuration(active_config)

    games = load_games_from_pgn(
        pgn_string
    )

    total_games = len(games)

    total_plies = count_total_plies(
        games
    )

    if total_plies < 1:
        raise ValueError(
            "The games do not contain any moves."
        )

    engine_path = find_stockfish(
        active_config["stockfish_path"]
    )

    print("=" * 70)
    print("CHESS ENGINE ANALYSIS")
    print("=" * 70)
    print(f"Games:                 {total_games}")
    print(f"Total player moves:    {total_plies}")

    print(
        "Stockfish depth:       "
        f"{active_config['depth']}"
    )

    print(
        "Candidate moves:       "
        f"{active_config['analysed_moves']}"
    )

    print(
        "Maximum move score:    "
        f"{active_config['analysed_moves']}"
    )

    print(
        "Blunder threshold:     "
        f"{active_config['blunder_threshold_cp']} cp"
    )

    print(
        f"Stockfish executable:  {engine_path}"
    )

    print("=" * 70)

    all_rows = []
    engine = None
    progress_bar = None

    analysis_start_time = (
        time.perf_counter()
    )

    completed_plies = 0
    detected_blunders = 0

    try:
        engine = (
            chess.engine.SimpleEngine.popen_uci(
                engine_path
            )
        )

        engine.configure(
            {
                "Threads": active_config[
                    "threads"
                ],
                "Hash": active_config[
                    "hash_size_mb"
                ],
            }
        )

        if active_config[
            "show_progress_bar"
        ]:
            progress_bar = tqdm(
                total=total_plies,
                desc="Overall analysis",
                unit="ply",
            )

        for game_number, game in enumerate(
            games,
            start=1
        ):
            white_name = game.headers.get(
                "White",
                "White"
            )

            black_name = game.headers.get(
                "Black",
                "Black"
            )

            event_name = game.headers.get(
                "Event",
                f"Game {game_number}"
            )

            game_result = game.headers.get(
                "Result",
                "*"
            )

            game_ply_count = sum(
                1
                for _ in game.mainline_moves()
            )

            game_message = (
                f"Starting game {game_number} "
                f"of {total_games}: "
                f"{white_name} vs {black_name} "
                f"({game_ply_count} plies)"
            )

            if progress_bar is not None:
                progress_bar.write(
                    game_message
                )
            else:
                print(game_message)

            board = game.board()

            for ply_number, played_move in enumerate(
                game.mainline_moves(),
                start=1
            ):
                move_start_time = (
                    time.perf_counter()
                )

                full_move_number = (
                    board.fullmove_number
                )

                mover_colour = board.turn

                if mover_colour == chess.WHITE:
                    colour_name = "White"
                    player_name = white_name
                else:
                    colour_name = "Black"
                    player_name = black_name

                played_san = board.san(
                    played_move
                )

                played_uci = (
                    played_move.uci()
                )

                fen_before = board.fen()

                current_move_label = (
                    build_move_label(
                        full_move_number,
                        colour_name,
                        played_san,
                    )
                )

                if (
                    progress_bar is not None
                    and active_config[
                        "show_current_move"
                    ]
                ):
                    progress_bar.set_postfix_str(
                        f"Game {game_number}: "
                        f"{player_name} "
                        f"{current_move_label}"
                    )

                analysis = analyse_single_move(
                    engine,
                    board,
                    played_move,
                    active_config["depth"],
                    active_config[
                        "analysed_moves"
                    ],
                    active_config[
                        "blunder_threshold_cp"
                    ],
                )

                if analysis["is_blunder"]:
                    detected_blunders += 1

                move_analysis_seconds = (
                    time.perf_counter()
                    - move_start_time
                )

                maximum_score = active_config[
                    "analysed_moves"
                ]

                accuracy_percent = (
                    analysis["accuracy_score"]
                    / maximum_score
                    * 100
                )

                row = {
                    "game_number": game_number,
                    "event": event_name,
                    "result": game_result,
                    "white": white_name,
                    "black": black_name,
                    "ply": ply_number,
                    "move_number": (
                        full_move_number
                    ),
                    "colour": colour_name,
                    "player": player_name,
                    "played_move_san": (
                        played_san
                    ),
                    "played_move_uci": (
                        played_uci
                    ),
                    "move_label": (
                        current_move_label
                    ),
                    "accuracy_score": analysis[
                        "accuracy_score"
                    ],
                    "maximum_accuracy_score": (
                        maximum_score
                    ),
                    "accuracy_percent": (
                        accuracy_percent
                    ),
                    "engine_rank": analysis[
                        "engine_rank"
                    ],
                    "is_in_analysed_moves": (
                        analysis[
                            "is_in_analysed_moves"
                        ]
                    ),
                    "is_blunder": analysis[
                        "is_blunder"
                    ],
                    "evaluation_loss_cp": (
                        analysis[
                            "evaluation_loss_cp"
                        ]
                    ),
                    "evaluation_loss_pawns": (
                        analysis[
                            "evaluation_loss_cp"
                        ]
                        / 100
                    ),
                    "best_move_san": analysis[
                        "best_move_san"
                    ],
                    "best_move_uci": analysis[
                        "best_move_uci"
                    ],
                    "best_evaluation": analysis[
                        "best_evaluation"
                    ],
                    "played_evaluation": (
                        analysis[
                            "played_evaluation"
                        ]
                    ),
                    "best_score_cp": analysis[
                        "best_score_cp"
                    ],
                    "played_score_cp": analysis[
                        "played_score_cp"
                    ],
                    "analysis_seconds": (
                        move_analysis_seconds
                    ),
                    "fen_before_move": (
                        fen_before
                    ),
                }

                add_candidate_columns(
                    row,
                    analysis["ranked_moves"],
                    active_config[
                        "analysed_moves"
                    ],
                )

                all_rows.append(row)

                board.push(
                    played_move
                )

                completed_plies += 1

                elapsed_seconds = (
                    time.perf_counter()
                    - analysis_start_time
                )

                average_seconds_per_ply = (
                    elapsed_seconds
                    / completed_plies
                )

                remaining_plies = (
                    total_plies
                    - completed_plies
                )

                estimated_remaining_seconds = (
                    average_seconds_per_ply
                    * remaining_plies
                )

                if progress_bar is not None:
                    progress_bar.update(1)

                update_interval = active_config[
                    "progress_update_every_plies"
                ]

                must_print_update = (
                    completed_plies
                    % update_interval
                    == 0
                    or completed_plies
                    == total_plies
                )

                if must_print_update:
                    elapsed_text = (
                        format_elapsed_time(
                            elapsed_seconds
                        )
                    )

                    remaining_text = (
                        format_elapsed_time(
                            estimated_remaining_seconds
                        )
                    )

                    progress_message = (
                        f"Progress: "
                        f"{completed_plies}/"
                        f"{total_plies} plies | "
                        f"Elapsed {elapsed_text} | "
                        f"Estimated remaining "
                        f"{remaining_text} | "
                        f"Blunders "
                        f"{detected_blunders} | "
                        f"Last move "
                        f"{current_move_label} "
                        f"took "
                        f"{move_analysis_seconds:.2f}s"
                    )

                    if progress_bar is not None:
                        progress_bar.write(
                            progress_message
                        )
                    else:
                        print(
                            progress_message
                        )

            game_complete_message = (
                f"Completed game "
                f"{game_number} "
                f"of {total_games}: "
                f"{white_name} vs "
                f"{black_name}"
            )

            if progress_bar is not None:
                progress_bar.write(
                    game_complete_message
                )
            else:
                print(
                    game_complete_message
                )

    finally:
        if progress_bar is not None:
            progress_bar.close()

        if engine is not None:
            engine.quit()

    if not all_rows:
        raise ValueError(
            "The PGN did not contain analysable moves."
        )

    move_results = pd.DataFrame(
        all_rows
    )

    decimal_places = active_config[
        "round_decimal_places"
    ]

    numeric_columns_to_round = [
        "accuracy_percent",
        "evaluation_loss_pawns",
        "analysis_seconds",
    ]

    move_results[
        numeric_columns_to_round
    ] = move_results[
        numeric_columns_to_round
    ].round(decimal_places)

    game_summary = create_game_summary(
        move_results,
        active_config["analysed_moves"],
    )

    total_elapsed_seconds = (
        time.perf_counter()
        - analysis_start_time
    )

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(
        f"Games analysed:       {total_games}"
    )
    print(
        f"Player moves:         {completed_plies}"
    )
    print(
        f"Blunders detected:    {detected_blunders}"
    )
    print(
        "Total analysis time: "
        f"{format_elapsed_time(total_elapsed_seconds)}"
    )
    print(
        "Average time per ply: "
        f"{total_elapsed_seconds / completed_plies:.2f} "
        "seconds"
    )

    if active_config[
        "automatically_save_csv"
    ]:
        detailed_filename = active_config[
            "detailed_csv_filename"
        ]

        summary_filename = active_config[
            "summary_csv_filename"
        ]

        move_results.to_csv(
            detailed_filename,
            index=False
        )

        game_summary.to_csv(
            summary_filename,
            index=False
        )

        print(
            f"Detailed results:     "
            f"{detailed_filename}"
        )

        print(
            f"Summary results:      "
            f"{summary_filename}"
        )

    print("=" * 70)

    if active_config["show_graphs"]:
        plot_all_games(
            move_results,
            active_config,
        )

    return move_results, game_summary

move_results, game_summary = analyse_pgn(
    pgn_string=pgn_text
)

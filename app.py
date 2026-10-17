%%writefile app.py
import streamlit as st
import chess
import chess.engine
import chess.pgn
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import shutil

# Configure Streamlit page layout
st.set_page_config(page_title="Chess PGN Analyzer", layout="wide")
st.title("♟️ Automated Chess PGN Engine Analyzer")
st.write("Enter a PGN game below to run a deep analysis of every move using the Stockfish chess engine.")

MATE_SCORE = 100000

# Helper functions adapted for Streamlit
def find_stockfish():
    possible_paths = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        shutil.which("stockfish")
    ]
    for path in possible_paths:
        if path and os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None

def score_to_cp(score, colour):
    numerical_score = score.pov(colour).score(mate_score=MATE_SCORE)
    return int(numerical_score) if numerical_score is not None else 0

def score_description(score, colour):
    pov_score = score.pov(colour)
    if pov_score.is_mate():
        mate_distance = pov_score.mate()
        return f"Mate in {mate_distance}" if mate_distance else "Mate"
    centipawn_score = pov_score.score()
    return f"{centipawn_score / 100:+.2f}" if centipawn_score is not None else "0.00"

def calculate_accuracy_score(engine_rank, analysed_moves):
    if engine_rank is None or engine_rank < 1 or engine_rank > analysed_moves:
        return 0
    return analysed_moves + 1 - engine_rank

def analyse_single_move(engine, board, played_move, depth=12, analysed_moves=20):
    mover_colour = board.turn
    legal_move_count = board.legal_moves.count()
    requested_lines = min(analysed_moves, legal_move_count)

    engine_result = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=requested_lines)
    if isinstance(engine_result, dict):
        engine_lines = [engine_result]
    else:
        engine_lines = engine_result

    ranked_moves = []
    played_move_line = None

    for returned_rank, info in enumerate(engine_lines, start=1):
        principal_variation = info.get("pv", [])
        returned_score = info.get("score")
        if not principal_variation or returned_score is None:
            continue
        suggested_move = principal_variation[0]
        line_data = {
            "rank": returned_rank,
            "move": suggested_move,
            "score_cp": score_to_cp(returned_score, mover_colour),
            "evaluation": score_description(returned_score, mover_colour),
            "san": board.san(suggested_move)
        }
        ranked_moves.append(line_data)
        if suggested_move == played_move:
            played_move_line = line_data

    if played_move_line is None:
        played_analysis = engine.analyse(board, chess.engine.Limit(depth=depth), root_moves=[played_move])
        played_score_object = played_analysis.get("score")
        played_score_cp = score_to_cp(played_score_object, mover_colour) if played_score_object is not None else 0
        played_evaluation = score_description(played_score_object, mover_colour) if played_score_object is not None else "Unknown"
        engine_rank = None
    else:
        played_score_cp = played_move_line["score_cp"]
        played_evaluation = played_move_line["evaluation"]
        engine_rank = played_move_line["rank"]

    best_line = ranked_moves[0]
    best_score_cp = best_line["score_cp"]
    accuracy_score = calculate_accuracy_score(engine_rank, analysed_moves)
    evaluation_loss_cp = max(0, best_score_cp - played_score_cp)
    is_blunder = evaluation_loss_cp >= 300

    return {
        "accuracy_score": accuracy_score,
        "engine_rank": engine_rank,
        "is_in_analysed_moves": (engine_rank is not None),
        "is_blunder": is_blunder,
        "evaluation_loss_cp": evaluation_loss_cp,
        "best_move_san": best_line["san"],
        "best_evaluation": best_line["evaluation"],
        "played_evaluation": played_evaluation,
        "best_score_cp": best_score_cp,
        "played_score_cp": played_score_cp,
    }

# User Input section
pgn_input = st.text_area(
    "Paste PGN here:",
    height=250,
    placeholder="[Event \"Live Chess\"]...\n\n1. e4 c5 2. Nf3..."
)

run_analysis = st.button("Run Stockfish Analysis", type="primary")

if run_analysis:
    if not pgn_input.strip():
        st.warning("Please enter a valid PGN first.")
    else:
        engine_path = find_stockfish()
        if not engine_path:
            st.error("Stockfish engine not found on the server. Please install it on your server using 'sudo apt-get install stockfish'.")
        else:
            # Process PGN
            pgn_stream = io.StringIO(pgn_input)
            game = chess.pgn.read_game(pgn_stream)

            if game is None:
                st.error("Failed to parse PGN. Check formatting.")
            else:
                white_player = game.headers.get("White", "White")
                black_player = game.headers.get("Black", "Black")
                event_name = game.headers.get("Event", "Live Chess")
                st.info(f"Analyzing game: **{white_player}** vs **{black_player}**")

                board = game.board()
                moves_to_analyze = list(game.mainline_moves())
                total_moves = len(moves_to_analyze)

                progress_bar = st.progress(0)
                status_text = st.empty()

                rows = []

                # Start Stockfish
                engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                try:
                    for idx, played_move in enumerate(moves_to_analyze):
                        status_text.text(f"Analyzing move {idx+1}/{total_moves}...")

                        full_move_number = board.fullmove_number
                        mover = "White" if board.turn == chess.WHITE else "Black"
                        played_san = board.san(played_move)

                        analysis = analyse_single_move(engine, board, played_move)

                        rows.append({
                            "game_number": 1,
                            "white": white_player,
                            "black": black_player,
                            "event": event_name,
                            "move_number": full_move_number,
                            "colour": mover,
                            "Move": f"{full_move_number}. {played_san}" if mover == "White" else f"{full_move_number}... {played_san}",
                            "Player": white_player if mover == "White" else black_player,
                            "Played Move": played_san,
                            "Best Move": analysis["best_move_san"],
                            "Played Evaluation": analysis["played_evaluation"],
                            "Best Evaluation": analysis["best_evaluation"],
                            "accuracy_score": analysis["accuracy_score"],
                            "evaluation_loss_pawns": analysis["evaluation_loss_cp"] / 100,
                            "Blunder": "🚨 BLUNDER" if analysis["is_blunder"] else "OK",
                            "is_blunder": analysis["is_blunder"]
                        })

                        board.push(played_move)
                        progress_bar.progress((idx + 1) / total_moves)

                finally:
                    engine.quit()

                status_text.success("Analysis Complete!")

                df_results = pd.DataFrame(rows)

                # Match Summaries / Statistics
                white_blunders = len(df_results[(df_results["colour"] == "White") & (df_results["is_blunder"] == True)])
                black_blunders = len(df_results[(df_results["colour"] == "Black") & (df_results["is_blunder"] == True)])
                white_avg_score = df_results[df_results["colour"] == "White"]["accuracy_score"].mean()
                black_avg_score = df_results[df_results["colour"] == "Black"]["accuracy_score"].mean()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric(label=f"{white_player} (White) Blunders", value=white_blunders)
                col2.metric(label=f"{black_player} (Black) Blunders", value=black_blunders)
                col3.metric(label=f"{white_player} Avg Accuracy Score", value=f"{white_avg_score:.2f} / 20")
                col4.metric(label=f"{black_player} Avg Accuracy Score", value=f"{black_avg_score:.2f} / 20")

                # Recreating the high quality graph exactly like the Google Colab version
                st.subheader("Move Accuracy Plot")
                fig, ax = plt.subplots(figsize=(12, 5))

                white_moves = df_results[df_results["colour"] == "White"]
                black_moves = df_results[df_results["colour"] == "Black"]

                if not white_moves.empty:
                    ax.plot(white_moves["move_number"], white_moves["accuracy_score"],
                            marker="o", linewidth=1.5, label=f"White: {white_player}", color="#1f77b4")
                if not black_moves.empty:
                    ax.plot(black_moves["move_number"], black_moves["accuracy_score"],
                            marker="s", linewidth=1.5, label=f"Black: {black_player}", color="#ff7f0e")

                # Blunder Annotations
                blunders = df_results[df_results["is_blunder"] == True]
                if not blunders.empty:
                    ax.scatter(blunders["move_number"], blunders["accuracy_score"],
                               color="red", marker="x", s=150, linewidths=3, label="Blunder", zorder=10)

                    for _, row in blunders.iterrows():
                        move_prefix = f"{row['move_number']}." if row["colour"] == "White" else f"{row['move_number']}..."
                        annotation = f"{move_prefix}{row['Played Move']}\nLoss: {row['evaluation_loss_pawns']:.2f}"
                        ax.annotate(annotation, (row["move_number"], row["accuracy_score"]),
                                    xytext=(5, 10), textcoords="offset points", fontsize=8, color="red")

                ax.set_title(f"{white_player} vs {black_player} - {event_name}")
                ax.set_xlabel("Game move number")
                ax.set_ylabel("Move score")
                ax.set_ylim(-0.35, 20.35)
                ax.set_yticks(range(0, 21, 2))
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

                st.subheader("Detailed Move-by-Move Log")
                st.dataframe(df_results[["Move", "Player", "Played Move", "Best Move", "Played Evaluation", "Best Evaluation", "Blunder"]], use_container_width=True)

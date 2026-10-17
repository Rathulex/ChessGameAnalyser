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

def analyse_single_move(engine, board, played_move, depth=12):
    mover_colour = board.turn
    engine_result = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
    
    # Best move analysis
    best_info = engine_result[0]
    best_move = best_info.get("pv", [played_move])[0]
    best_score_cp = score_to_cp(best_info.get("score"), mover_colour)
    best_eval_desc = score_description(best_info.get("score"), mover_colour)
    
    # Played move analysis
    played_analysis = engine.analyse(board, chess.engine.Limit(depth=depth), root_moves=[played_move])
    played_score_cp = score_to_cp(played_analysis.get("score"), mover_colour)
    played_eval_desc = score_description(played_analysis.get("score"), mover_colour)
    
    eval_loss = max(0, best_score_cp - played_score_cp)
    is_blunder = eval_loss >= 300
    
    return {
        "is_blunder": is_blunder,
        "evaluation_loss_cp": eval_loss,
        "best_move_san": board.san(best_move),
        "best_evaluation": best_eval_desc,
        "played_evaluation": played_eval_desc,
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
                        
                        mover = "White" if board.turn == chess.WHITE else "Black"
                        played_san = board.san(played_move)
                        
                        analysis = analyse_single_move(engine, board, played_move)
                        
                        rows.append({
                            "Move": f"{(idx//2)+1}. {played_san}" if mover == "White" else f"{(idx//2)+1}... {played_san}",
                            "Player": white_player if mover == "White" else black_player,
                            "Played Move": played_san,
                            "Best Move": analysis["best_move_san"],
                            "Played Evaluation": analysis["played_evaluation"],
                            "Best Evaluation": analysis["best_evaluation"],
                            "Centipawn Loss": analysis["evaluation_loss_cp"],
                            "Blunder": "🚨 BLUNDER" if analysis["is_blunder"] else "OK"
                        })
                        
                        board.push(played_move)
                        progress_bar.progress((idx + 1) / total_moves)
                        
                finally:
                    engine.quit()
                
                status_text.success("Analysis Complete!")
                
                df_results = pd.DataFrame(rows)
                
                # Visualizations / Columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Match Summary")
                    # Summary Calculations
                    white_blunders = len(df_results[(df_results["Player"] == white_player) & (df_results["Blunder"] == "🚨 BLUNDER")])
                    black_blunders = len(df_results[(df_results["Player"] == black_player) & (df_results["Blunder"] == "🚨 BLUNDER")])
                    
                    st.metric(label=f"{white_player} (White) Blunders", value=white_blunders)
                    st.metric(label=f"{black_player} (Black) Blunders", value=black_blunders)
                
                with col2:
                    st.subheader("Centipawn Loss Progression")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(df_results[df_results["Player"] == white_player].index, 
                            df_results[df_results["Player"] == white_player]["Centipawn Loss"], 
                            label=f"{white_player} Loss", color="blue")
                    ax.plot(df_results[df_results["Player"] == black_player].index, 
                            df_results[df_results["Player"] == black_player]["Centipawn Loss"], 
                            label=f"{black_player} Loss", color="orange")
                    ax.set_ylabel("Centipawn Loss")
                    ax.legend()
                    st.pyplot(fig)
                
                st.subheader("Move-by-Move Log")
                st.dataframe(df_results, use_container_width=True)

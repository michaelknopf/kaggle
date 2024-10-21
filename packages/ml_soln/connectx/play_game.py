import argparse
import webbrowser
from ml_soln.connectx import ctx
from ml_soln.connectx.agent import model_agent

def run_game(job_name, opponent):
    # Load the model and training state
    model = ctx().model_persistence.load_model(job_name)

    # Create the agent
    agent = model_agent(model)

    # Determine the opponent
    if opponent == "random":
        opponent_agent = "random"
    elif opponent == "model":
        opponent_agent = model_agent(model)
    else:
        # Default to negamax
        opponent_agent = "negamax"

    # Run the game between the agent and the opponent
    _ = ctx().kaggle_env.run([agent, opponent_agent])

    # Render the game to an HTML file
    html = ctx().kaggle_env.render(mode="html")
    html_path = ctx().paths.clone(job_name).model_dir / '..' / 'game.html'
    html_path = html_path.resolve().absolute()
    with open(html_path, 'w') as f:
        f.write(html)

    # Open the game in a web browser
    webbrowser.open_new_tab(f'file://{html_path}')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a ConnectX game between a model and an opponent.")

    # Required positional argument for job name
    parser.add_argument('job_name', type=str, help='The name of the job (model identifier) to load.')

    # Optional argument for the opponent, with "negamax" as the default
    parser.add_argument('-o', '--opponent',
                        type=str,
                        choices=['negamax', 'random', 'model'],
                        default='negamax',
                        help='Choose the opponent: "negamax", "random", or "model". Default is "negamax".')

    # Parse the arguments
    args = parser.parse_args()

    # Run the game with the given job name and opponent
    run_game(args.job_name, args.opponent)

if __name__ == "__main__":
    main()

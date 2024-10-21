import logging
from dataclasses import dataclass

import numpy as np
from keras import Model
from rich import progress as pg

from ml_soln.common.sagemaker_utils import sm_utils
from ml_soln.connectx import ctx
from ml_soln.connectx.dqn import DQN, Experience
from ml_soln.connectx.stubs import ConnectXObservation

logger = logging.getLogger(__name__)

@dataclass
class TrainState:

    all_total_rewards: np.ndarray
    all_avg_rewards: np.ndarray     # Last 100 steps
    all_epsilons: np.ndarray


class Trainer:

    def __init__(self):
        self.iteration = 0

    def train(self,
              train_model: Model = None,
              target_model: Model = None):

        if not train_model:
            train_model = ctx().model.model
        if not target_model:
            target_model = ctx().model.new_model()

        train_dqn: DQN = DQN(train_model)
        target_dqn: DQN = DQN(target_model)

        ts = TrainState(all_total_rewards=np.empty(ctx().hyperparams.episodes),
                        all_avg_rewards=np.empty(ctx().hyperparams.episodes),
                        all_epsilons=np.empty(ctx().hyperparams.episodes))

        epsilon = ctx().hyperparams.start_epsilon

        # Initialize the rich progress bar
        if not sm_utils.is_sagemaker:
            progress = pg.Progress(
                pg.TextColumn("[bold blue]{task.description}"),
                pg.BarColumn(),
                pg.TextColumn("[green]Episode"),
                pg.MofNCompleteColumn(),
                pg.TaskProgressColumn(),
                pg.TextColumn("[cyan]Time elapsed: "),
                pg.TimeElapsedColumn(),
                pg.TextColumn("[cyan]Time remaining: "),
                pg.TimeRemainingColumn(),
                pg.TextColumn("[bold blue]Average Reward: {task.fields[avg_reward]:>3.1f}"),
                pg.TextColumn("[bold green]Episode Reward: {task.fields[episode_reward]:>2.0f}"),
                pg.TextColumn("[bold red]Epsilon: {task.fields[epsilon]:>3.3f}"),
            )
            task = progress.add_task('Training',
                                     total=ctx().hyperparams.episodes,
                                     avg_reward=0,
                                     episode_reward=0,
                                     epsilon=epsilon)
            progress.start()
        else:
            progress = None
            task = None

        for n in range(ctx().hyperparams.episodes):
            epsilon = max(ctx().hyperparams.min_epsilon,
                          epsilon * ctx().hyperparams.decay)

            if sm_utils.is_sagemaker:
                logger.info(f'Starting game {n}')

            # play an episode
            total_reward = self.play_game(train_dqn, target_dqn, epsilon)

            ts.all_total_rewards[n] = total_reward
            avg_reward = ts.all_total_rewards[max(0, n - 100):(n + 1)].mean()
            ts.all_avg_rewards[n] = avg_reward
            ts.all_epsilons[n] = epsilon

            if not sm_utils.is_sagemaker:
                progress.update(task,
                                advance=1,
                                episode_reward=total_reward,
                                avg_reward=avg_reward,
                                epsilon=epsilon)

        if not sm_utils.is_sagemaker:
            progress.stop()

        return ts

    def play_game(self,
                  train_dqn: DQN,
                  target_dqn: DQN,
                  epsilon: float):

        rewards = 0
        done = False

        trainer = ctx().connect_x_gym.new_trainer()
        observation: ConnectXObservation = trainer.reset()
        self.fix_observation(observation)

        while not done:
            # Using epsilon-greedy to get an action
            action = train_dqn.get_action(observation, epsilon)

            # Caching the information of current state
            prev_observation = observation

            # Take action
            observation, reward, done, agent_info = trainer.step(action)
            self.fix_observation(observation)

            # Apply new rules
            if done:
                # Win
                if reward == 1:
                    reward = 20
                # Loss
                elif reward == 0:
                    reward = -20
                # Draw
                else:
                    reward = 10
            else:
                # Penalize longer games
                reward = -0.05

            rewards += reward

            # Adding experience into buffer
            exp = Experience(state=prev_observation,
                             action=action,
                             reward=reward,
                             next_state=observation,
                             done=done)
            train_dqn.add_experience(exp)

            # Train the training model by using experiences in buffer and the target model
            train_dqn.train(target_dqn)
            self.iteration += 1
            if self.iteration % ctx().hyperparams.copy_step == 0:
                # Update the weights of the target model when reaching enough "copy step"
                target_dqn.copy_weights(train_dqn)

        return rewards

    @staticmethod
    def fix_observation(observation: ConnectXObservation):
        """
        Kaggle env uses this "struct" object that tries to make a dict behave like an object.
        Due to a bug, it is not working correctly when our agent is the 2nd player.
        This patches the issue.
        """
        observation.__dict__['board'] = observation['board']
        observation.__dict__['mark'] = observation['mark']

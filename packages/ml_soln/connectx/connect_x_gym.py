from typing import List, Tuple

from kaggle_environments.helpers import Observation, Configuration
from kaggle_environments.utils import Struct


class ConnectXObservation(Observation, Struct):

    def __init__(self,
                 mark: int,
                 board: List[int]):
        super().__init__()
        # The current serialized Board (rows x columns).
        self.board = board
        # Which player the agent is playing as (1 or 2).
        self.mark = mark

class ConnectXConfiguration(Configuration, Struct):

    def __init__(self,
                 columns: int,
                 rows: int,
                 inarow: int):
        super().__init__()
        self.columns = columns
        self.rows = rows
        self.inarow = inarow

class AgentInfo(Struct):
    pass

class KaggleTrainer:

    # agent.observation, reward, agent.status != "ACTIVE", agent.info
    def step(self, action: str) -> Tuple[Observation, float, bool, AgentInfo]:
        pass

    def reset(self) -> Observation:
        pass

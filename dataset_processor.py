import pandas as pd
import random
from typing import List, Tuple, Dict

class ShakespeareDatasetProcessor:
    """
    Class that processes the CSV and splits the data in train-test sets.
    """
    def __init__(self, csv_path: str, test_split: float = 0.1, dev_split: float = 0.1, random_seed: int = 42):
        self.csv_path = csv_path
        self.test_split = test_split
        self.dev_split = dev_split
        self.random_seed = random_seed
        self.data = None
        self.train_samples = []
        self.dev_samples = []
        self.eval_samples = []


    def load_data(self) -> None:
        """The function loads the CSV file into a DataFrame.
        """
        self.data = pd.read_csv(self.csv_path)


    def process_data(self) -> List[Tuple[str, str]]:
        """
        The function processes the data to extract a list of (Player, PlayerLine) tuples.

        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple contains a speaker and their spoken line.
        """
        # Extract only the Player and PlayerLine columns
        relevant_data = self.data[['Player', 'PlayerLine']]
        relevant_data = relevant_data.dropna(subset=['Player', 'PlayerLine'])

        # Add ':' to separate the player from the line
        samples = [(f"{player}: ", line) for player, line in relevant_data.itertuples(index=False, name=None)]

        return samples


    def split_data(self, samples: List[Tuple[str, str]]) -> None:
        """
        Splits the data into training and evaluation sets.

        Args:
            samples (List[Tuple[str, str]]): The list of all (Player, PlayerLine) tuples.
        """
        random.seed(self.random_seed)
        random.shuffle(samples)

        split_idx1 = int(len(samples) * (1 - self.test_split - self.dev_split))
        split_idx2 = int(len(samples) * (1 - self.test_split))
        self.train_samples = samples[:split_idx1]
        self.dev_samples = samples[split_idx1:split_idx2]
        self.eval_samples = samples[split_idx2:]

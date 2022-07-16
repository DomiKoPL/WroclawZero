import torch
from torch.utils.data import Dataset
from torch import tensor
import gc


class GameSamplesDataset(Dataset):
    __slots__ = "state_memory", "legal_moves_memory", "policy_memory", "value_memory", "generations",

    def __init__(self) -> None:
        self.state_memory: tensor = torch.zeros(0)
        self.legal_moves_memory: tensor = torch.zeros(0)
        self.policy_memory: tensor = torch.zeros(0)
        self.value_memory: tensor = torch.empty(0)

        self.generations: list[tuple[int, int, int]] = []
        """
            (generation number, begin, size)
        """

    def __len__(self) -> int:
        return self.state_memory.shape[0]

    def __getitem__(self, index: int) -> tuple[tensor, tensor, tensor, tensor]:
        return (self.state_memory[index], self.legal_moves_memory[index], self.policy_memory[index], self.value_memory[index])

    def generations_count(self) -> int:
        return len(self.generations)

    def add_generation(self, generation: int, states: tensor, legal_moves: tensor, policies: tensor, values: tensor) -> None:
        self.generations.append((generation, len(self), states.shape[0]))
        self.state_memory = torch.cat((self.state_memory, states), dim=0)
        self.legal_moves_memory = torch.cat((self.legal_moves_memory, legal_moves), dim=0)
        self.policy_memory = torch.cat((self.policy_memory, policies), dim=0)
        self.value_memory = torch.cat((self.value_memory, values), dim=0)

    def remove_generations_before(self, generation: int) -> None:
        """
            removes generations before <generation> and update self.generations list
        """

        to_erase = len([1 for g, _, _ in self.generations if g < generation])

        if to_erase == 0:
            return

        _, begin, size = self.generations[to_erase - 1]
        del self.generations[0:to_erase]

        erase_size = begin + size
        self.state_memory = self.state_memory[erase_size:]
        self.legal_moves_memory = self.legal_moves_memory[erase_size:]
        self.policy_memory = self.policy_memory[erase_size:]
        self.value_memory = self.value_memory[erase_size:]

        self.generations = [(g, begin - erase_size, size)
                            for g, begin, size in self.generations]

        gc.collect()

    def debug(self) -> None:
        print(
            f"{self.state_memory.shape } {self.legal_moves_memory.shape } {self.policy_memory.shape } {self.value_memory.shape }")
        print(f"{len(self) = }, {self.generations = }")
        print(f"{self.generations_count() = }")


def main():
    data = GameSamplesDataset()

    def add(gen: int, cnt: int):
        data.add_generation(gen, torch.ones((cnt, 5)), torch.ones((cnt, 3)),
                            torch.ones((cnt, 3)), torch.ones((cnt, 1)))

    data.debug()

    for i in range(7):
        add(i, 5 + i * 5)
        data.debug()

    data.remove_generations_before(4)

    data.debug()

    data.remove_generations_before(10)

    data.debug()


if __name__ == "__main__":
    main()

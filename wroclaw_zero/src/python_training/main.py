import subprocess
import sys
from traceback import print_tb
import torch
from GameSamplesDataset import GameSamplesDataset
from CommunicationInterface import CommunicationInterface
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
import torch.optim as optim
import os
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def parse_model(config: dict):
    layers = nn.Sequential()

    for layer_config in config:
        if type(layer_config) != dict:
            print(f"Unknown type {layer_config}")
            exit(1)

        if layer_config["type"] == "Input":
            # This layers is for C++ only
            continue

        if layer_config["type"] == "Linear":
            layer = nn.Linear(layer_config["input"], layer_config["output"])
        elif layer_config["type"] == "Conv2d":
            layer = nn.Conv2d(
                layer_config["in_channels"], layer_config["out_channels"], kernel_size=layer_config["kernel"],
                stride=layer_config["stride"], padding=layer_config["padding"])

        layers.append(layer)

        if "activation" in layer_config:
            layer = eval("nn." + layer_config["activation"])()
            layers.append(layer)

        # print(layer)

    print(layers)

    return layers


class NeuralNetwork(nn.Module):
    def __init__(self, config: dict):
        super(NeuralNetwork, self).__init__()
        self.base = parse_model(config)

        self.value_head = nn.Sequential(
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
        )

    def forward(self, x, legal_moves):
        x = self.base(x)

        policy = self.policy_head(x[:, 1:]).reshape((x.size()[0], -1))
        # print(f"{policy = }")
        # print(f"{legal_moves = }")
        policy = torch.mul(policy, legal_moves)
        policy += torch.mul(1 - legal_moves, -99999999)
        # print(f"{policy = }")

        value = self.value_head(x[:, 0]).reshape((x.size()[0], -1))

        return policy, value


def save_model(model: nn.Sequential) -> bytes:
    model_bytes = bytes()
    total_bytes = 0
    model.eval()
    for name, param in model.named_parameters():
        w = param.cpu().detach().numpy()
        # ? nn_avx stores Linear layer as [#input, #output]
        # ? where PyTorch stores as [#output, #input]

        if len(param.shape) == 2:
            w = w.T

        b = w.tobytes()
        model_bytes += b
        total_bytes += len(b)

        # print(f"Total bytes = {total_bytes}")
    return model_bytes


def load_model(model: nn.Sequential, file):
    with torch.no_grad():
        for _, param in model.named_parameters():
            w = param.cpu().detach().numpy()

            b = w.tobytes()
            bytes_to_load = file.read(len(b))

            array = np.frombuffer(bytes_to_load, dtype='float32')

            if len(param.shape) == 2:
                # (X, Y) -> (Y, X) -> bajty
                array = array.reshape(param.shape[1], param.shape[0])
                array = array.T
                param.data = nn.parameter.Parameter(torch.tensor(array))
            elif len(param.shape) == 4:
                array = array.reshape(param.shape)
                param.data = nn.parameter.Parameter(torch.tensor(array))
            else:
                param.data = nn.parameter.Parameter(torch.tensor(array))


class Trainer:
    def __init__(self, communication_interface: CommunicationInterface, config: dict):
        self.communication_interface = communication_interface
        self.config = config
        self.writer = SummaryWriter(
            os.path.join(config["data_path"], "logs"))
        config = config["learning"]
        self.max_memory_size = config["max_memory_size"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.lr = float(config["lr"])
        self.weight_decay = float(config["weight_decay"])
        self.dataset: GameSamplesDataset = GameSamplesDataset()
        self.generation = 0
        self.models_stats_path = os.path.join(
            self.config["data_path"], "models_stats.yaml")

    def add_samples(self, msg: tuple[int, int, int, int, bytes]):
        _, samples_cnt, state_len, policy_len, samples = msg

        input_shape = (state_len, )
        if self.config["model"][0]["type"] == "Input":
            input_shape = self.config["model"][0]["shape"]

        print(f"[PYTHON] Samples input shape {input_shape}")

        cnt = samples_cnt * state_len * 4
        states = np.frombuffer(samples[0:cnt], dtype="float32")

        states = torch.tensor(states.reshape(samples_cnt, *input_shape))
        print(f"[PYTHON] {states.shape = }")

        start = cnt

        cnt = samples_cnt * policy_len * 4
        legal_moves = np.frombuffer(samples[start:start+cnt], dtype="int32")
        legal_moves = torch.tensor(
            legal_moves.reshape(samples_cnt, policy_len))
        print(f"[PYTHON] {legal_moves.shape = }")
        start += cnt

        cnt = samples_cnt * policy_len * 4
        policies = np.frombuffer(samples[start:start+cnt], dtype="float32")
        policies = torch.tensor(policies.reshape(samples_cnt, policy_len))
        print(f"[PYTHON] {policies.shape = }")
        start += cnt

        cnt = samples_cnt * 4
        values = np.frombuffer(
            samples[start:start+cnt], dtype="float32")
        values = torch.tensor(values.reshape(samples_cnt, 1))
        print(f"[PYTHON] {values.shape = }")

        self.dataset.add_generation(
            self.generation, states, legal_moves, policies, values)

    def check_previous_generations(self):
        print(f"{self.communication_interface.models_directory = }")

        max_gen = 0
        for file in os.listdir(self.communication_interface.models_directory):
            gen = int(file.split("_")[1])
            max_gen = max(max_gen, gen)

        print(f"{max_gen = }")

        if max_gen > 0:
            self.generation = max_gen + 1

    def training_loop(self):
        self.init_model()
        self.check_previous_generations()

        current_state = 0

        while True:
            if current_state == 0:
                self.communication_interface.request_self_play(self.generation)
                current_state = 1
            elif current_state == 2:
                self.generation += 1

                model = self.train_model()
                network_bytes: bytes = save_model(model)

                self.communication_interface.save_model(
                    self.generation, network_bytes)

                self.communication_interface.request_model_comparison(
                    self.generation, network_bytes)

                current_state = 3

            msg = self.communication_interface.receive_message()
            if msg is None:
                continue

            msg_type = msg[0]

            if current_state == 1:
                if msg_type == 4:
                    self.add_samples(msg)
                    current_state = 2
                    continue

            if current_state == 3:
                if msg_type == 1:
                    current_state = 0
                    continue

            if msg_type == 2:
                _, tag, scalar, step = msg
                self.writer.add_scalar(tag, scalar, step)
                self.writer.flush()
            elif msg_type == 3:
                _, main_tag, tag_scalar_dict, step = msg
                self.writer.add_scalars(main_tag, tag_scalar_dict, step)
                self.writer.flush()

    def init_model(self):
        if os.path.exists(self.models_stats_path):
            return

        print(f"Creating random model.")
        model = NeuralNetwork(self.config["model"])

        model_bytes: bytes = save_model(model)

        self.communication_interface.save_model(0, model_bytes)

        with open(self.models_stats_path, "w") as file:
            models_stats = {
                "best1": 0
            }

            yaml.dump(models_stats, file)

    def train_model(self):
        print(f'Training start\n')

        def get_lr():
            if self.generation == 1:
                return 6 * self.lr
            if self.generation <= 5:
                return 4 * self.lr
            if self.generation <= 10:
                return 2 * self.lr

            return self.lr

        model = NeuralNetwork(self.config["model"])

        with open(self.models_stats_path, "r") as file:
            models_stats = yaml.safe_load(file)
            best_generation = models_stats["best1"]

        print(f"Loading best model {best_generation = }.")

        load_model(model, open(os.path.join(
            self.communication_interface.models_directory, f"model_{best_generation}"), "rb"))

        model.to('cuda')
        model.train()

        # TODO: add value of clip to config
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer = optim.AdamW(
            model.parameters(), lr=get_lr(), weight_decay=self.weight_decay)

        mse_loss = MSELoss()
        entropy_loss = CrossEntropyLoss()

        mem_size = 4
        if self.generation >= 5:
            mem_size += (self.generation - 5) // 2

        mem_size = min(mem_size, self.max_memory_size)

        self.dataset.remove_generations_before(self.generation - mem_size)
        print(f"{self.generation = }")
        self.dataset.debug()

        batch_size = self.batch_size

        print(f"[PYTHON] {len(self.dataset) = }")

        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.writer.add_scalar("Training/Learning rate",
                               get_lr(), self.generation)

        self.writer.add_scalar(
            "Training/Samples", len(self.dataset), self.generation)

        self.writer.add_scalar(
            "Training/Window size", self.dataset.generations_count(), self.generation)

        epochs: int = min(self.generation, self.epochs)

        self.writer.add_scalar(
            "Training/Epochs", epochs, self.generation)

        self.writer.flush()

        for epoch in range(epochs):  # loop over the dataset multiple times
            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0

            for i, data in enumerate(dataloader):
                game_states, legal_moves, policies, values = data
                game_states = game_states.cuda()
                legal_moves = legal_moves.cuda()
                policies = policies.cuda()
                values = values.cuda()

                optimizer.zero_grad()

                policy_outputs, value_outputs = model(game_states, legal_moves)

                value_loss = mse_loss(value_outputs, values)
                policy_loss = entropy_loss(policy_outputs, policies)

                loss = value_loss + policy_loss

                loss.backward()
                optimizer.step()

                epoch_value_loss += value_loss.item() * policy_outputs.shape[0]
                epoch_policy_loss += policy_loss.item() * \
                    policy_outputs.shape[0]

            self.writer.add_scalar("Loss/value", epoch_value_loss /
                                   len(self.dataset), self.generation * self.epochs + epoch)
            self.writer.add_scalar("Loss/policy", epoch_policy_loss /
                                   len(self.dataset), self.generation * self.epochs + epoch)

            self.writer.flush()

            print(f'[{epoch + 1}] value_loss: {epoch_value_loss / len(self.dataset):.7f} policy_loss: {epoch_policy_loss / len(self.dataset):.7f} ')

        print('Finished Training')



        return model


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Invalid number of arguments!\nYou need to pass config!")
        exit(1)

    with open(sys.argv[1], "r") as file:
        config: dict = yaml.safe_load(file)

    model = NeuralNetwork(config["model"])

    communication_interface: CommunicationInterface = CommunicationInterface(
        subprocess.Popen(["build/bin/main", sys.argv[1]],
                         stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                         stderr=sys.stderr, preexec_fn=os.setsid),
        os.path.join(config["data_path"], "archive"),
        os.path.join(config["data_path"], "models"),
    )

    trainer = Trainer(communication_interface, config)
    trainer.training_loop()

    # from pympler.tracker import SummaryTracker
    # tracker = SummaryTracker()

    # from guppy import hpy; h=hpy()

    # dataset = GameSamplesDataset(20_000, 130)
    # for i in range(10):
    #     samples = communication_interface.receive_samples()
    #     dataset.add_samples(samples)
    # # tracker.print_diff()
    # # print(dataset.memory[0])
    # print(h.heap())

    # for i in range(4):
    #     print(h.heap().byid[0].sp)
    #     print('\n\n\n')

    # import sys
    # print(len(dataset.memory), sys.getsizeof(dataset.memory))
    # # 14144240
    # print(total_size(dataset.memory[0].state), total_size(dataset.memory[0].state.storage()), dataset.memory[0].state.size())
    # print(total_size(dataset.memory[0].policy), total_size(dataset.memory[0].policy.storage()), dataset.memory[0].policy.size())
    # print(total_size(dataset.memory[0].value), total_size(dataset.memory[0].value.storage()), dataset.memory[0].value.size())

    # print(dataset.state_memory[:3], '\n\n')
    # print(dataset.policy_memory[:3], '\n\n')
    # print(dataset.value_memory[:3])

    # # print(len(dataset.memory))
    # # sleep(10)
    # for x, y, z in DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn):
    #     print(x, y, z)
    #     break

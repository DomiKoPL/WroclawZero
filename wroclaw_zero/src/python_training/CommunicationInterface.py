import os
import subprocess
import atexit
import numpy as np


class CommunicationInterface():
    cpp_process: subprocess.Popen[bytes] = None
    models_directory: str = None

    def __init__(self, cpp_process: subprocess.Popen[bytes],
                 archive_directory: str, models_directory: str):
        self.cpp_process = cpp_process
        self.archive_directory = archive_directory
        self.models_directory = models_directory

        # https://itecnote.com/tecnote/python-killing-child-process-when-parent-crashes-in-python/
        atexit.register(self.cpp_process.terminate)

        if not os.path.exists(self.archive_directory):
            os.makedirs(self.archive_directory)

        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)

    def __to_bytes(self, x: int) -> bytes:
        return int.to_bytes(x, 4, "little")

    def __from_bytes(self, b: bytes) -> int:
        return int.from_bytes(b, "little")

    def send_message(self, type: int, content: bytes) -> None:
        msg = self.__to_bytes(type) + content
        self.cpp_process.stdin.write(msg)
        self.cpp_process.stdin.flush()

    def request_self_play(self, generation: int) -> None:
        print(f"[PYTHON] Requesting self play {generation = }")
        self.send_message(0, self.__to_bytes(generation))

    def request_model_comparison(self, generation: int, network_bytes: bytes) -> None:
        print(f"[PYTHON] Requesting model comparison")

        content: bytes = self.__to_bytes(
            generation) + self.__to_bytes(len(network_bytes)) + network_bytes

        self.send_message(1, content)

    def save_model(self, generation: int, network_bytes: bytes) -> None:
        with open(os.path.join(self.models_directory, f"model_{generation}"), "wb") as file:
            file.write(network_bytes)

    def receive_message(self) -> None | tuple[int, bytes] | tuple[int, int, int, int, bytes] | tuple[int, str, float, int] | tuple[int, str, dict, int]:
        msg_type: int = self.__from_bytes(self.cpp_process.stdout.read(4))
        # print(f"[PYTHON] {msg_type = }")

        if msg_type == 0:
            raise Exception("msg_type is 0")

        if msg_type == 1:
            return (msg_type, b"")

        if msg_type == 4:
            # samples
            print(f"[PYTHON] reading samples")

            samples_cnt: int = self.__from_bytes(
                self.cpp_process.stdout.read(4))
            state_len: int = self.__from_bytes(self.cpp_process.stdout.read(4))
            policy_len: int = self.__from_bytes(
                self.cpp_process.stdout.read(4))

            # state_len  - state
            # policy_len - legal_moves
            # policy_len - policy
            # 1          - value
            samples_len: int = state_len + 2 * policy_len + 1

            print(f"[PYTHON] {samples_cnt = } {state_len = } {policy_len = }")
            samples: bytes = self.cpp_process.stdout.read(
                samples_cnt * samples_len * 4)

            return (msg_type, samples_cnt, state_len, policy_len, samples)

        if msg_type == 2:
            # scalar for TensorBoard
            tag_len: int = self.__from_bytes(self.cpp_process.stdout.read(4))
            tag: str = self.cpp_process.stdout.read(tag_len).decode("utf-8")
            scalar: float = np.frombuffer(
                self.cpp_process.stdout.read(4), dtype="float32")[0]
            step: int = self.__from_bytes(self.cpp_process.stdout.read(4))
            return (msg_type, tag, scalar, step)

        if msg_type == 3:
            # scalars for TensorBoard
            main_tag_len: int = self.__from_bytes(
                self.cpp_process.stdout.read(4))
            main_tag: str = self.cpp_process.stdout.read(
                main_tag_len).decode("utf-8")
            tags_count: int = self.__from_bytes(
                self.cpp_process.stdout.read(4))
            tag_scalar_dict = dict()

            for _ in range(tags_count):
                tag_len: int = self.__from_bytes(
                    self.cpp_process.stdout.read(4))
                tag: str = self.cpp_process.stdout.read(
                    tag_len).decode("utf-8")
                scalar: float = np.frombuffer(
                    self.cpp_process.stdout.read(4), dtype="float32")[0]

                tag_scalar_dict[tag] = scalar

            step: int = self.__from_bytes(self.cpp_process.stdout.read(4))

            # print(f"{main_tag = }")
            # print(f"{tag_scalar_dict = }")
            # print(f"{step = }")
            return (msg_type, main_tag, tag_scalar_dict, step)

        return None

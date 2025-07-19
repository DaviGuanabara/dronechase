from abc import ABC, abstractmethod


class Sensor(ABC):
    def __init__(self, parent_id: int, client_id: int):
        self.parent_id = parent_id
        self.client_id = client_id

    @abstractmethod
    def read_data(self):
        pass

    @abstractmethod
    def update_data(self):
        pass

    @abstractmethod
    def reset(self):
        pass

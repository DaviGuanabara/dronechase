from abc import ABC, abstractmethod


class ActuatorInterface(ABC):
    @abstractmethod
    def apply(self, *args, **kwargs) -> None:
        """Apply the actuator's primary function."""
        pass


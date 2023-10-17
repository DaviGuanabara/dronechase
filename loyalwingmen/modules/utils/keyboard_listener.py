from typing import Optional, Dict, Union
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import numpy as np


class KeyboardListener:
    def __init__(self, keymap: Dict[Union[Key, KeyCode], list]):
        self.key_map = keymap
        self.key: Optional[Union[Key, KeyCode]] = None
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key: Union[Key, KeyCode, None]):
        if key in self.key_map:
            self.key = key
        else:
            self.key = None

    def on_release(self, key: Union[Key, KeyCode, None]):
        self.key = None
        if key == keyboard.Key.esc:
            self.listener.stop()

    def get_action(self) -> np.ndarray:
        if self.key is not None:
            action = self.key_map[self.key]
        else:
            action = self.key_map["default"] # type: ignore
        
        return np.array(action)

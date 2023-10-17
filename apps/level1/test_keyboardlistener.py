import pytest
from pynput.keyboard import Key, KeyCode
from loyalwingmen.modules.utils.keyboard_listener import KeyboardListener
import pytest
import pyautogui
import time


class TestKeyboardListener:
    @pytest.fixture
    def valid_keymap(self):
        # Assuming the actions are lists of floats
        return {
            Key.space: [1.0, 2.0, 3.0],  # Using a pynput Key
            KeyCode.from_char("a"): [0.5, 0.5, 0.5],  # Using a KeyCode
            # "default": [0.0, 0.0, 0.0]  # Commented out to avoid str as key
        }

    def test_keyboard_listener_initialization(self, valid_keymap):
        listener = KeyboardListener(valid_keymap)
        assert listener.key is None
        # ... further assertions and tests as needed.

    def test_keyboard_listener_get_action_with_valid_key(self, valid_keymap):
        listener = KeyboardListener(valid_keymap)

        # Give 1 second delay before we simulate a key press
        time.sleep(5)

        # Simulate pressing the 'space' key
        pyautogui.press("space")

        # Give 1 second delay for the listener to react to the simulated key press
        time.sleep(1)

        action = listener.get_action()
        # Compare the returned action with the one you expect for Key.space
        assert list(action) == valid_keymap[Key.space]
        # ... further assertions and tests as needed.

    # Add more test methods within the class as needed.

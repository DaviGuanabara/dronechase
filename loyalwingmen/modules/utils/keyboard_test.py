from pynput import keyboard


class KeyboardListener:
    def __init__(self):
        self.collect_events()
        self.k = "{0}".format("0")

    def on_press(self, key):
        try:
            # print("alphanumeric key {0} pressed".format(key.char))
            self.k = "{0}".format(key.char)
        except AttributeError:
            # print("special key {0} pressed".format(key))
            self.k = "{0}".format(key)

    def on_release(self, key):
        # print("{0} released".format(key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False

        self.k = "{0}".format("0")

    def collect_events(self):
        # Collect events until released
        # with keyboard.Listener(
        #    on_press=self.on_press, on_release=self.on_release
        # ) as listener:
        # listener.join()

        # ...or, in a non-blocking fashion:
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

    def get_action(self):
        return self.k


kl = KeyboardListener()
while True:
    k = kl.get_action()
    print(k)

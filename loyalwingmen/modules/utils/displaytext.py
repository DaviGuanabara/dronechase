import time
import curses
import numpy as np


def format_array_to_string(arr, keywords, string=""):
    foo = ["{:.4f}".format(m) for m in arr]

    index_foo = 0

    for keyword, quantity in keywords:
        values = ""
        for i in range(quantity):
            if i == quantity - 1:
                values += foo[index_foo + i] + "\n"
            else:
                values += foo[index_foo + i] + ", "

        index_foo = index_foo + quantity
        string += keyword + ": " + values

    return string


def format_returns(obs, reward, action):
    keywords_obs = [
        ("drone_position", 3),
        ("drone_velocity", 3),
        ("cube_position", 3),
        ("cube_velocity", 3),
        ("direction", 3),
        ("distance", 1),
    ]

    keywords_action = [("Action (Velocity Vector):", 4)]

    string = "Reward:" + "{:.4f}".format(reward) + " \n"
    string = format_array_to_string(obs, keywords_obs, string)
    string = format_array_to_string(action, keywords_action, string)

    return string


def log(string):
    stdscr = curses.initscr()
    stdscr.addstr(0, 0, string)
    stdscr.refresh()


def log_returns(obs, reward=0, action=[0, 0, 0, 0]):
    log(format_returns(obs, reward, action))


# test2()

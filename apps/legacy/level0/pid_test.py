import pytest

import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)

from loyalwingmen.modules.environments.level0.nova_controladora import PID


class TestPID:
    def ziegler_nichols_tuning(self, Ku, Tu):
        return {
            "P": (0.5 * Ku, 0, 0),
            "PI": (0.45 * Ku, 1.2 * 0.45 * Ku / Tu, 0),
            "PID": (0.6 * Ku, 2 * 0.6 * Ku / Tu, 0.6 * Ku * Tu / 8),
        }["PID"]

    def pid_test(self, controller, desired_value, dt):
        current_value = 0
        current_value_rate = 0

        print("desired_value:", desired_value, "current_value:", current_value)
        history = []
        for _ in range(20):
            output = controller.compute(
                desired_value, current_value, current_value_rate, dt
            )

            print(
                "current_value:",
                current_value,
                "output:",
                output,
            )
            current_value += output
            current_value_rate = output

            history.append(current_value)

        return current_value

    def calculate_Ku_Tu(self, final_values):
        # After some tests, I found that the results bellow
        Ku = 2
        Tu = 2
        return Ku, Tu

    def test_PID_autotuning(self):
        desired_value = 1
        dt = 1
        kp, ki, kd = 2, 0, 0
        controller = PID(kp, ki, kd)

        final_values = []

        for i in range(2):
            final_value = self.pid_test(controller, desired_value, dt)

            final_values.append(final_value)
            print(f"final_value_{i}: {final_value} - kp: {kp} - ki: {ki} - kd: {kd}")

            Ku, Tu = self.calculate_Ku_Tu(final_values)
            kp, ki, kd = self.ziegler_nichols_tuning(Ku, Tu)
            controller.update_gains(kp, ki, kd)
            controller.reset()

        assert (
            abs(final_values[-1] - desired_value) < 0.1
        )  # Verifica se o valor final é próximo de desired_value


if __name__ == "__main__":
    pytest.main([__file__])

import os
import sys
import logging


class DirectoryManager:
    @staticmethod
    def find_base_dir(
        current_path: str, target_dir: str = "loyalwingmen", debug: bool = True
    ) -> str:
        if debug:
            logging.basicConfig(level=logging.DEBUG)

        max_iterations = 5

        for iterations in range(max_iterations):
            current_path, base_dir = os.path.split(current_path)

            logging.debug(f"Current path: {current_path}")
            logging.debug(f"Base dir: {base_dir}")

            if base_dir == target_dir:
                path = os.path.join(current_path, base_dir)
                logging.debug(f"path: {os.path.join(current_path, base_dir)}")
                return path

            has_reached_root = not current_path or current_path == os.path.sep
            has_reached_max_interations = iterations == max_iterations - 1
            if has_reached_root or has_reached_max_interations:
                return os.path.dirname(os.path.abspath(__file__))
        # This line should not be reached, but it's here as a safety measure
        logging.warning("Unexpected termination of loop.")
        return ""

    @staticmethod
    def get_base_dir(debug: bool = False) -> str:
        path = os.path.abspath(__file__)
        base_dir = DirectoryManager.find_base_dir(path)
        logging.error("Could not find base dir") if not base_dir and debug else None

        return base_dir

    @staticmethod
    def get_outputs_dir(app_name: str, debug: bool = False) -> str:
        base_dir = DirectoryManager.get_base_dir()
        outputs_dir = os.path.join(base_dir, "outputs", app_name)
        logging.debug(
            f"(get_outputs_dir) outputs_dir: {outputs_dir}"
        ) if debug else None

        DirectoryManager.create_directory(outputs_dir)

        return outputs_dir

    @staticmethod
    def get_logs_dir(app_name: str, debug: bool = False) -> str:
        outputs_dir = DirectoryManager.get_outputs_dir(app_name, debug)
        logs_dir = os.path.join(outputs_dir, "logs")
        DirectoryManager.create_directory(logs_dir)
        return logs_dir

    @staticmethod
    def get_models_dir(app_name: str, debug: bool = False) -> str:
        outputs_dir = DirectoryManager.get_outputs_dir(app_name, debug)
        models_dir = os.path.join(outputs_dir, "models")
        DirectoryManager.create_directory(models_dir)
        return models_dir

    @staticmethod
    def create_directory(directory_path: str) -> str:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return directory_path

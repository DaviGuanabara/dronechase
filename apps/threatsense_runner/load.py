


from core.rl_framework.utils.io_data import IOData



if __name__ == "__main__":

    folder_path = "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\collect_and_save"
    io_data = IOData(folder_path)
    loader = io_data.get_loader(batch_size=128, shuffle=True)

    for obs, target in loader:
        print(obs)
        print(target)
        break
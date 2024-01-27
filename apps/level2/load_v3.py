import sys

sys.path.append("..")
from stable_baselines3 import PPO

# from loyalwingmen.environments.level2.pyflyt_level2_environment import (
#    PyflytL2Enviroment as Level2,
# )

from loyalwingmen.environments.level2.pyflyt_level2_environment_modified_v3 import (
    PyflytL2EnviromentModifiedV3 as Level2,
)


def setup_environment():
    print("Load - Level 2")
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output_mac\\baysian_optimizer_app\\level2_1.00M_03.12.2023_baysian\\models_dir\\h[512, 256, 128]-f15-lr0.001\\mPPO-r4985.2099609375-sd1149.3851318359375.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v3\\20_12.2023_level3_1.00M_v2_for_extractor_v3\\models_dir\\h[512, 1024, 128]_f15_lr0.0001\\t0_PPO_r1362.58.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v3\\21_12.2023_level3_1.00M_v2_for_extractor_v3\\Trial_12\\models_dir\\h[256, 512, 512]_f15_lr0.001\\t12_PPO_r1469.09.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output_mac\\baysian_optimizer_app\\level2_1.00M_03.12.2023_baysian\\models_dir\\h[512, 256, 128]-f15-lr0.001\\mPPO-r4985.2099609375-sd1149.3851318359375.zip"

    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single\\05_01_2024_level2_2.00M_single\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r1911.27.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single\\05_01_2024_level2_2.00M_single\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r1320.40.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2\\08_01_2024_level2_2.00M_v2\\models_dir\\h[512, 128, 128]_f15_lr0.0001\\t0_PPO_r1482.58.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single\\09_01_2024_level2_2.00M_single\\Trial_4\\models_dir\\best_model.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single\\09_01_2024_level2_2.00M_single\\Trial_4\\models_dir\\h[512, 256, 128]_f15_lr0.001"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single\\09_01_2024_level2_2.00M_single\\Trial_3\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r1451.10.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single\\09_01_2024_level2_2.00M_single\\Trial_4\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r1726.59.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\from_level_2\\mPPO-r4985.2099609375-sd1149.3851318359375.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single\\09_01_2024_level2_2.00M_single\\Trial_5\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r1516.86.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2\\08_01_2024_level2_2.00M_v2\\models_dir\\h[256, 128, 128]_f15_lr0.0001\\t0_PPO_r1190.40.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2\\08_01_2024_level2_2.00M_v2\\models_dir\\h[256, 512, 256]_f15_lr0.0001\\t0_PPO_r1289.85.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2\\08_01_2024_level2_2.00M_v2\\models_dir\\h[128, 128, 256]_f15_lr0.0001\\t0_PPO_r1411.96.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2\\08_01_2024_level2_2.00M_v2\\models_dir\\h[512, 512, 256]_f15_lr0.001\\t0_PPO_r105.61.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2\\08_01_2024_level2_2.00M_v2\\models_dir\\h[512, 128, 128]_f15_lr0.01\\t0_PPO_r-5922.54.zip"

    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_v_old\\level2_1.00M_10.01.2023_baysian_v_old_p2\\Trial_0\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r3462.26.zip"
    # model = PPO.load("./ppo_level2_lidar_03_12_2023")

    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_v_old_modified\\level2_1.00M_10.01.2023_baysian_v_old_p2\\Trial_2\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r931.11.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_modified\\10_01_2024_level2_1.00M_v2_modified_ultimo\\Trial_0\\models_dir\\h[512, 512, 256]_f15_lr0.001\\t0_PPO_r1427.73.zip"
    # TRIAL 10
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single_modified\\12_01_2024_level2_1.00M_single_modified\\Trial_10\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r1472.57.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single_modified\\12_01_2024_level2_1.00M_single_modified\\Trial_21\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r1271.07.zip"

    path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single_modified\\12_01_2024_level2_4.00M_single_modified_for_chosen_one\\Trial_0\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r-5809.20.zip"
    # path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_single_modified\\12_01_2024_level2_4.00M_single_modified_for_chosen_one\\Trial_2\\models_dir\\best_model.zip"
    path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_optimizer_app_v2_modified\\13_01_2024_level2_1.00M_v2_modified_v2\\Trial_0\\models_dir\\h[1024, 256, 128]_f15_lr0.001\\t0_PPO_r-1506.75.zip"
    path = "/Users/Davi/Documents/GitHub/PyFlyt/apps/level2/output/baysian_optimizer_app_v2_single_modified/13_01_2024_level2_1.00M_single_modified_v2_p2/Trial_0/models_dir/h[512, 256, 128]_f15_lr0.01/t0_PPO_r-496.96.zip"

    path = "/Users/Davi/Documents/GitHub/PyFlyt/apps/level2/output/baysian_optimizer_app_v2_single_modified/13_01_2024_level2_1.00M_single_modified_v2_p3/Trial_0/models_dir/h[512, 256, 128]_f15_lr0.001/t0_PPO_r2048.09.zip"

    path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_v_old_modified\\level2_1.00M_14.01.2023_baysian_v_old_p1\\Trial_0\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r1936.98.zip"
    path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_v_old_modified\\level2_2.00M_14.01.2023_baysian_v_old_p2\\Trial_0\\models_dir\\h[512, 256, 128]_f15_lr0.001\\t0_PPO_r2340.12.zip"
    path = "C:\\Users\\davi_\\Documents\\GitHub\\PyFlyt\\apps\\level2\\output\\baysian_v_old_modified_v3\\level2_2.00M_14.01.2023_baysian_v_old_modified_v3_multi_topology_p3_homeoffice\\Trial_0\\models_dir\\h[128, 256, 256]_f15_lr0.001\\t0_PPO_r2675.07.zip"
    env = Level2(GUI=True, rl_frequency=15)
    model = PPO.load(path, env)

    # env = Level2(GUI=True, rl_frequency=15)
    observation, _ = env.reset(0)
    print(observation.keys())
    return env, model, observation


def main():
    env, model, observation = setup_environment()

    for steps in range(50_000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # logging.debug(f"(main) reward: {reward}")
        print(f"reward:{reward:.2f} - action:{action} - observation:{observation}")
        # print(f"reward:{reward:.2f} - action:{action}")

        if terminated:
            print("terminated")
            observation, info = env.reset(0)


if __name__ == "__main__":
    main()

# test_student.py
import numpy as np
import torch
from threatsense.level5.level5_fusion_environment import Level5FusionEnvironment as Level5Fusion



def format_obs(obs, device):
    """
    Converte a observação do ambiente para o formato esperado pelo modelo.
    Assume que obs é uma lista de dicts (um por drone) e usa o primeiro (RL agent).
    """
    if isinstance(obs, list):
        obs = obs[0]  # usa o RL agent (primeiro pursuer)
    return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
            for k, v in obs.items()}


def run_student(model_path, episodes=5, max_steps=500, gui=True):
    # === Inicializa ambiente ===
    env = Level5Fusion(GUI=gui)
    observation_space = env.observation_space

    # === Carrega modelo treinado ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentWithFLIA(observation_space).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"✅ Modelo carregado de {model_path}")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        while not done and step < max_steps:

            # ação aleatória para last_action
            # obs["last_action"] = np.random.uniform(-1, 1, size=(4,)).astype(np.float32)
            print(obs)

            obs_torch = format_obs(obs, device)

            with torch.no_grad():
                action = model(obs_torch).cpu().numpy()[0]

            print(action)
            print(f"Step {step+1}: Action taken: {action}")

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            total_reward += reward
            step += 1

        print(
            f"[Episódio {ep+1}] Reward total: {total_reward:.3f} | Steps: {step}")

    env.close()
    print("🚀 Teste concluído.")


if __name__ == "__main__":
    model_path = (
        # "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\trained_models_all\\models\\student_train_epoch5_5.pt"
        # "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\07.09.2025_trained_models_all\\models\\student_last_model_0.pt"
        #"C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\09.09.2025_trained_models_all_3\\models\\student_last_model_0.pt"
        "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\output\\sl\\12.09.2025\\train_sl_9\\model.zip"
    )
    run_student(model_path, episodes=5, max_steps=500, gui=True)
# folder_path = "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\09.09.2025_trained_ppo_1\\student_ppo_final.zip"

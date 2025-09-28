from threatsense.level5.level5_eval_2bt_environment import Level52BTEvaluationEnvironment as Level5_2BTEval_Environment
import sys
import os
import numpy as np
import pandas as pd
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

grand_parent_directory = os.path.dirname(parent_directory)
sys.path.append(grand_parent_directory)


# Diretório de saída
output_dir = os.path.join(current_directory, "output", "evaluation", "2bt")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "results_2bt.xlsx")

# Inicializa o ambiente
env = Level5_2BTEval_Environment(GUI=False, rl_frequency=15)

N_EPISODES = 100
results = []

start_time = time.time()

for ep in range(N_EPISODES):
    observation, info = env.reset()
    terminated = False

    # roda até terminar o episódio
    while not terminated:
        action = np.zeros(4,)  # ação nula
        observation, reward, terminated, truncated, info = env.step(action)

    # fim do episódio → coleta resultados
    kills_dict = info["kills_per_drone"]

    ep_result = {v["name"]: v["kills"] for v in kills_dict.values()}
    ep_result["total_kills"] = sum(v["kills"] for v in kills_dict.values())
    results.append(ep_result)

    # estima tempo restante
    elapsed = time.time() - start_time
    avg_time = elapsed / (ep + 1)
    remaining = avg_time * (N_EPISODES - (ep + 1))

    print(
        f"Rodando episódio {ep+1}/{N_EPISODES} | "
        f"Tempo decorrido: {elapsed:.1f}s | "
        f"Estimado restante: {remaining:.1f}s",
        end="\r",
        flush=True
    )

print("\n✅ Avaliação concluída.")

# Converte para DataFrame
df = pd.DataFrame(results)

# Estatísticas
df_stats = pd.DataFrame({
    "mean": df.mean(),
    "std": df.std()
})

# Salva em Excel
with pd.ExcelWriter(output_file) as writer:
    df.to_excel(writer, sheet_name="raw_results", index=False)
    df_stats.to_excel(writer, sheet_name="summary_stats")

print(f"\n📊 Resultados salvos em: {output_file}")

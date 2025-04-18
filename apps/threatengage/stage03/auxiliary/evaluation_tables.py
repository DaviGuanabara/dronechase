import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Paths para os arquivos Excel
exp_01_path = Path("evaluation_03.02.2024_exp01_1bt_episode_info.xlsx")
exp_02_path = Path("evaluation_03.02.2024_exp02_1rl_episode_info.xlsx")
exp_03_path = Path("evaluation_03.02.2024_exp03_1rl_1bt_episode_info.xlsx")
exp_04_path = Path(
    "evaluation_03.02.2024_exp04_1rl_1bt_trained_stopped_episode_info.xlsx"
)
exp_05_path = Path("evaluation_03.02.2024_exp05_2RL_episode_info.xlsx")
exp_06_path = Path("evaluation_03.02.2024_exp06_2bt_episode_info.xlsx")
exp_07_path = Path("evaluation_04.02.2024_exp07_1RL_1NOT_MOVING_episode_info.xlsx")

paths = [
    exp_01_path,
    exp_02_path,
    exp_03_path,
    exp_04_path,
    exp_05_path,
    exp_06_path,
    exp_07_path,
]

dataframes = []
for i, csv_file_path in enumerate(paths):
    # Verifica se o arquivo existe
    if not csv_file_path.exists():
        print(f"Arquivo não encontrado: {csv_file_path}")
    else:
        # Lê o arquivo Excel, assumindo que a primeira planilha contém os dados e o cabeçalho está na primeira linha
        evaluation_data = pd.read_excel(csv_file_path, header=0)

        # Verifica se a coluna 'episode' existe
        if "episode" in evaluation_data.columns:
            evaluation_data.set_index("episode", inplace=True)
        else:
            print(f"Coluna 'episode' não existe no arquivo: {csv_file_path}")

        evaluation_data.insert(0, "exp", "exp" + str(i + 1))
        dataframes.append(evaluation_data)
        print(evaluation_data)

# Combina todos os DataFrames em um único DataFrame
combined_df = pd.concat(dataframes)

# Agrupando os dados por experimento e episódio
grouped_data = combined_df.groupby(["exp", "episode"])

# Calculando a soma de 'kills' para cada grupo
sums = grouped_data["kills"].sum().reset_index()

# Renomeando a coluna de soma para refletir o dado que ela representa
sums.columns = ["Experiment", "Episode", "Total Kills"]

# print(sums)

grouped_data = combined_df.groupby("exp")
# Calculando a média e o desvio padrão para a coluna 'kills' em cada grupo
stats = grouped_data["kills"].agg(["median", "mean", "std"]).reset_index()

# Renomeando as colunas para melhor clareza
stats.columns = ["Experiment", "Median Kills", "Mean Kills", "Standard Deviation"]

print(stats)

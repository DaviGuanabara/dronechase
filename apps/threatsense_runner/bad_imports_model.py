import zipfile
import json
import pickletools
import base64
import re
import pathlib

zip_path = pathlib.Path("C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\experiment_results\\threat_engage\\Etapa 03 - Level 4\\EXP03\\Treinamento - 1RL 1BT - EXP03\\31_01_2024_level4_2.00M_exp03_vFinal\\Trial_8\\models_dir\\h[128, 256, 512]_f15_lr0.0001\\t8_PPO_r6995.40.zip")
with zipfile.ZipFile(zip_path) as z:
    data = json.loads(z.read("data").decode())
    # estes campos guardam objetos pickled em base64
    b64_fields = [k for k, v in data.items() if isinstance(v, str)
                  and v.startswith("base64:")]
    for k in b64_fields:
        raw = base64.b64decode(data[k][7:])
        print(f"\n--- {k} ---")
        for op, arg, *_ in pickletools.genops(raw):
            if op.name == "GLOBAL":
                print("GLOBAL:", arg)  # mostra "modulo nomeobj"

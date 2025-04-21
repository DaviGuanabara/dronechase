# 📄 PROPOSTA DE REFATORAÇÃO DO Arquivo interactive_loader_plan.md

## 🎯 Objetivo
Refatorar os scripts `load.py` e `interactive.py` que estavam duplicados em múltiplos estágios do projeto (stage01 até stage04), consolidando-os em **um único script configurável**. A ideia é manter a clareza, facilitar manutenção e reduzir código redundante.

## 📦 Proposta de Arquitetura
Um único script `interactive.py`, parametrizável via CLI:

```bash
python interactive.py \
    --level 3 \
    --model_path "models/level3_best.zip" \
    --interactive
```

### 📥 Argumentos suportados:
| Argumento         | Tipo     | Descrição                                           |
|------------------|----------|----------------------------------------------------|
| `--level`        | int      | Define o estágio do ambiente (0, 1, 2, 3...)       |
| `--model_path`   | str      | Caminho para o modelo `.zip`                       |
| `--interactive`  | flag     | Se presente, ativa o controle manual do usuário    |


## 🔧 Lógica Interna do Script

```python
parser = argparse.ArgumentParser()
parser.add_argument("--level", type=int, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()

# 1. Criação do ambiente com base no nível
env = make_env(level=args.level, interactive=args.interactive)

# 2. Carregamento do modelo
model = PPO.load(args.model_path)

# 3. Execução do ambiente com ou sem controle manual
if args.interactive:
    run_user_controlled(env)
else:
    run_agent(env, model)
```

## ✅ Vantagens
- ✨ Clareza arquitetural
- ♻️ Redução de código duplicado
- 🔄 Facilidade de manutenção
- 📊 Pronto para testes visuais e análises comparativas por estágio

## 📌 Observações
- Esse plano não precisa ser implementado agora.
- Será útil para facilitar avaliações futuras e integração com pipelines de experimentação.
- Pode ser estendido com logging, gravação de vídeo, métricas, etc.




# Interactive Mode – Manual Testing

This folder contains scripts for manual control and visualization of the simulation.

- Purpose: To allow real-time testing of environment behavior and agent movement.
- Not used in training or evaluation stages.
- Control method: Keyboard (W/S/Arrow keys)

🧪 Useful for demonstration and debugging, but **not part of the experimental pipeline**.




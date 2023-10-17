QuadcopterController para Quadricópteros Crazy Fly (cf2x)

A QuadcopterController é uma controladora PID especializada projetada para o quadricóptero Crazy Fly (cf2x) e desempenha um papel fundamental na estabilização e controle do drone. Para operar eficazmente, a controladora requer uma série de dados e informações cruciais:

Essa controladora vai ser usada em uma simulação pybullet com um cf2x simulado, e os dados de entrada, o vetor velocidade, são de origem do SB3 PPO agente, que tem o objetivo de engajar um drone intruso, em um gymnasium ambiente.
Dados Essenciais:

Limites Operacionais e Especificações:
RPM Máximo (MAX_RPM): A rotação máxima permitida dos motores.
RPM de Pairamento (HOVER_RPM): RPM necessário para manter o drone pairando.
Velocidade Máxima (MAX_SPEED_KMH): A máxima velocidade que o drone pode atingir.
Constantes Físicas:
Massa (M): Peso total do drone.
Distância entre Motores (L): Distância entre os motores do quadricóptero.
Relação Força-Peso (THRUST2WEIGHT_RATIO): Relação entre a força gerada pelos motores e o peso total do quadricóptero.
Peso (WEIGHT): Peso total do quadricóptero.
Coeficientes e Parâmetros:
Coeficiente de Força dos Motores (KF): Determina a relação entre a força gerada pelos motores e a velocidade angular das hélices.
Coeficiente de Torque dos Motores (KM): Define o torque produzido pelos motores em relação à velocidade angular das hélices.
Coeficientes de Arrasto (DRAG_COEFF): Utilizados para modelar a resistência do ar.
Coeficientes Downwash (DW_COEFF_1, DW_COEFF_2, DW_COEFF_3): Coeficientes relacionados ao efeito downwash.
Coeficiente de Efeito Solo (GND_EFF_COEFF): Coeficiente que influencia o comportamento do drone próximo ao solo.
Raio das Hélices (PROP_RADIUS): O raio das hélices do quadricóptero.
Parâmetros de Colisão:
Altura (COLLISION_H): Parâmetro relacionado à altura.
Raio (COLLISION_R): Parâmetro relacionado ao raio.
Offset de Z (COLLISION_Z_OFFSET): Offset no eixo Z.
Momento de Inércia:
Matriz de Inércia (J) e sua Inversa (J_INV): Informações sobre o momento de inércia do drone em relação aos três eixos.
Dados Ambientais:
Aceleração devido à Gravidade (GRAVITY_ACCELERATION): Valor típico de aceleração devido à gravidade (aproximadamente 9.81 m/s²).
Características Principais:

A QuadcopterController oferece recursos essenciais para o controle eficaz do quadricóptero, incluindo:

PID Individual para Força e Torque: A controladora utiliza PID separados para controlar a força e o torque nas direções X, Y e Z, totalizando 18 parâmetros PID.
Anti-windup: Evita o acúmulo excessivo no termo integral do controlador PID.
Filtro Passa-Baixa para o Derivativo: Filtra ruídos no termo derivativo.
Zona Morta: Erros muito pequenos são considerados como zero.
Principais Funções:

A controladora desempenha as seguintes funções principais:

Controle de Atitude: Mantém a orientação desejada do quadricóptero usando informações de atitude, velocidade angular e aceleração angular do IMU.
Controle de Posição e Velocidade: Controla o movimento ao longo dos eixos X, Y e Z com base em dados de posição, velocidade e aceleração lineares do IMU.
Estabilização e Suavização: Minimiza erros e oscilações para proporcionar um voo suave e estável.
Feedback Contínuo: Opera em um ciclo de controle contínuo, recebendo dados em tempo real do IMU e calculando correções aos motores em alta frequência.
Parâmetros Cruciais:

Modelo Matemático do Drone:
A operação eficaz da controladora depende dos seguintes parâmetros cruciais:

Modelo Dinâmico: Um modelo dinâmico do quadricóptero é utilizado para calcular as respostas dos motores com base no vetor de velocidade desejado.
Feedback de Velocidade e Dados Inerciais: O feedback de velocidade proveniente do IMU, juntamente com os dados inerciais, é essencial para ajustar os motores com precisão.
Constantes PID Ajustáveis: As constantes PID (Kp, Ki, Kd) podem ser ajustadas para otimizar o desempenho do controlador.
Ajuste Online: Implemente um mecanismo de ajuste online que possa adaptar as constantes PID em tempo real com base no desempenho real do drone.
Com esses dados e recursos, a QuadcopterController garante um voo estável e controlado do quadricóptero no ambiente de simulação, priorizando a replicação precisa do vetor de velocidade desejado, mesmo em condições ideais sem perturbações externas.

A controladora desempenha as seguintes funções principais:

Controle de Atitude e Velocidade: Mantém a orientação desejada do quadricóptero e, ao mesmo tempo, ajusta a força de cada motor de forma dinâmica para alcançar o vetor de velocidade desejado. Isso é realizado utilizando informações de atitude, velocidade angular, aceleração angular e feedback de velocidade do IMU.

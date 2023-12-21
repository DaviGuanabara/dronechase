Dia 19/12/2023
Esse foi o melhor treinado no level 2. Não tenho os outros dados dele, e talvez eu possa melhorar ele. Enfim, vai ser melhorado aqui, no level 3

Vou colocar para o raio de nascimento dos invaders variarem de 3 a 6, e ver os resultados.
As informações dele são:

Hiddens: 512, 256, 128
Frequency: 15
Learning Rate: 0,001
bath size 256
Features Dim 512
Avg Score: 4985,21
Std Deviation: 1.149,385

1.000.000 steps.

###
No dia 20/10/2019, eu carreguei ele, e fiz um novo treinamento no level 3.
É importante perceber que o level 2 é mais simples. Ele bonifica pela proximidade com o alvo, e dá um bônus de 1000 pontos caso o alvo chega "pegue". O alvo é "pegue" quando a distância até o alvo é de 0.2.

No level 3, o alvo é "pegue" a uma distância de 1 metro. Porém, há o reload time. Durante o reload, a recompensa é por se afastar do alvo. Ela vai caíndo proporcionalmente ao tempo restante para o reload. Quando o reload termina, a recompensa volta a ser por se aproximar do alvo.

A distância que o alvo nasce e o agent nasce também aumentou. No level 2, a distância máxima era de 2 metros (O invader, ou kamikaze, começa em [0,0,0], já o pursuer começa em [1,1,1]. Depois o invader pode nascer em qualquer lugar entre [-1,-1,-1] e [1,1,1]. Sim, ainda não tem chão aqui).

No level 3, o invader pode nascer entre um raio de 2 até 6 metros da origem. O meu objetivo é que ele consiga nascer entre 10 e 20 metros da origem. Isso possibilitará treinar no level 4, no qual há ondas de invaders.

Assim, eu peguei o melhor da etapa passada, para treinar nessa etapa. 

O melhor até agora foi:
learning rate = 0.01
batch size = 256
2.000.000 steps.

O comportamento resultante até que é bom, mas precisa de mais treinamento. O agente se aproxima do alvo, mas por vezes, principalmente quando está longe, ele se perde.
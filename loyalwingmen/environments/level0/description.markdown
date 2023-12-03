Eu estou bem perto de realmente ter criado uma interface para comunicar o quadx com o meu ambiente.
Eu encapsulei o quadx completamente.

Mas há alguns pontos a considerar:
Primeiro: o state é um np.ndarray. Isso pode parecer inicialmente inofensivo, porém, é necessário sempre
ter um conhecimento prévio das posições de cada leitura, dificultando bastante qualquer alteração.

Segundo: o quadx.state usa velocidades angular e linear locais, e não globais. Isso é algo que dificulta bastante
qualquer comparação.

Seguindo pela minha experiência, é muito fácil quebrar algo aqui no código e eu ficar muito doido tentando entender o pq que quebrou
então vou tentar encapsular ao máximo para não precisa mexer nesse state. E sim, manter o que já tenho.

O meu imu funciona, e faz o lidar funcionar. isso é o que é importante.
Com o ambiente feito, vou conseguir fazer o level1: o loyalwingman (persuer) buscar o loitering munition (invader). Com esse level feito usando a física direitinho, pronto. Vou avançar pro Lidar com a física, e depois múltiplos drones com a física. Ao menos estou a um passo de avançar nisso. Não sei se vai dar tempo de fazer tudo até o final do ano, mas n tenho muitas outras alternativas tambêm.

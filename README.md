# Algoritomo de Seleção de Features: pBGSK-FS: Population based Binary Gaining–Sharing Knowledge-based Optimization

Algoritmo é baseado no desempenho de diferentes combinações de features em utilizando knn.
O processo de seleção de feature é reconhecidamente um problema np-completo, considerando n features temos O(n!) maneiras de combinar as diferentes features.Para solucionar isto existem diferentes metodos, este algoritmo é considerado um wrapper, ou seja é utilizado um modelo (knn) em uma parcela das features , este modelo nos retorna um score, detalhado mais abaixo, que norteia o processo de seleção de feature, portanto este metodo envolve um custo maior e consequentemente uma maior precisão.

## Diferencial

Um grande diferencia no setor da computação é a habilidade em copiar soluções, claro que adicionando algum nivel de abstração e engenhosidade, em relação a seleções de features temos inspirações de diversos fenomenos animais e processos, este artigo tenta abstrair o acumulo de conhecimento humano para um algoritmo.

## Como o artigo pensa sobre conhecimento e relações humanas.

Durante nossa vida temos diversos influencias , mas se pensarmos com atenção quais são as principais influencias em cada fase da vida ?
Durante a infancia a familia , vizinhos e professores uma rede social pequena e contida.Depois, ao fim da  adolescencia e começo da vida adulta adentramos em redes maiores, faculdade , professores universitarios , mercado de trabalho lentamente as nossas influencias positivas e negativas aumentam , comportamentos antes não tolerados no ambiente familiar podem ser tolerados fora dele, enquanto outros comportamentos antes aceitos e bem vistos não são mais bem vistos.

Em geral a ideia base é o ambiente é mais influente durante a infancia e durante a vida adulta influencias antes distantes agora são importantes.

## Como isto é aplicado.

Temos algums elementos.
1. População
2. Individuos
3. Dimensões
4. Senioriade

Cada individuo possui um vetor binario que indica quais dimensões estão presentes.
A População possui os individuos e quais dimensões são junior ou senior.

### Alem disto temos as fases
0. Population Creation
1. Population Classification 
2. Gain-Share Junior
3. Gain-Share Senior
4. Population Reduction

O loop básico é entre 1 a 4.

#### Population Creation 
Uma população de individuos com o numero de dimensões entre [k_lower, k_upper] é criado.
Um vetor com a senioriade de cada dimensão e estabelecido.
A cada geração esse vetor aumenta em um ritmo K, aumentando o numero de dimensões seniors.

####  Population Classification
Cada individuo é classificado em relação a (1- acuracia - features_usadas/features_totais)
Os individuos são ordenados em relação ao desempenho (menor melhor) 

#### Fase Junior

Durante as fases temos , basicamente 3 elementos Exemplo positivo , negativo e neutro.
Durante a fase junior os elementos imediatamente melhor e imediatamente pior (no vetor de desenpenho) servem de exemplo, individuos imediatamente e imediatamente pior.

* **$d$**: Dimensão total do problema (número original de atributos).
* **$NFE$**: Número atual de avaliações da função (*Number of Function Evaluations*).
* **$MaxNFE$**: Número máximo de avaliações permitidas.
* **$K$**: Taxa de conhecimento (fator de experiência gerado aleatoriamente).
* **$P$**: % (float 0-1) da partiçao(p melhores , p piores e 1-2p medianos)

Pseudo codigo do artigo numero 1
* np são os individuos
* d são as dimensões.
* f() é o vetor de desenpenho
* xtk e a dimensão k do individuo x
```pseudo
for t = 1 : Np do
    for k = 1 : d do
        if rand ≤ kR (knowledge ratio) then
            if f(xt) > f(xR) then
                xtk_new = xt + kf * [(xt-1 - xt+1) + (xR - xt)]
            else
                xtk_new = xt + kf * [(xt-1 - xt+1) + (xt - xR)]
            end if
        else
            xtk_new = xtk_old
        end if
    end for
end for
```
### Resultados da Fase *Beginners-Intermediate* (Caso 1)

Esta tabela representa os resultados possíveis durante o estágio de ganhos e partilhas para iniciantes-intermediários (*Junior Stage*), especificamente para o **Caso 1**, onde a aptidão do indivíduo é melhor que a do indivíduo selecionado para compartilhar: $f(x_t) > f(x_R)$ Esta tabela representa os resultados possíveis durante o estágio de ganhos e partilhas para iniciantes-intermediários (*Junior Stage*), especificamente para o **Caso 1**, onde a aptidão do indivíduo atual é melhor que a do indivíduo selecionado aleatoriamente para compartilhar: $f(x_t) > f(x_R)$

No algoritmo original contínuo, a equação de atualização para este cenário é definida como:

$$x_{tk}^{new} = x_t + k_f \cdot [(x_{t-1} - x_{t+1}) + (x_R - x_t)]$$

Temos dois subcasos.
* **Subcaso (a):** Se o valor de $x_{t-1}$ for **igual** ao de $x_{t+1}$, o resultado binário final modificado será sempre igual ao valor de $x_R$.
* **Subcaso (b):** Quando $x_{t-1}$ for **diferente** de $x_{t+1}$, o resultado binário final modificado será igual ao valor de $x_{t-1}$. Para manter o limite do espaço discreto, valores calculados fora do domínio binário são ajustados pela heurística: `2` é convertido para `1`, e `-1` é convertido para `0`.


| Subcaso | $x_{t-1}$ | $x_{t+1}$ | $x_R$ | Resultado (Cálculo) | Resultado Modificado (Binário) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Subcaso (a)** | 0 | 0 | 0 | 0 | **0** |
| | 0 | 0 | 1 | 1 | **1** |
| | 1 | 1 | 0 | 0 | **0** |
| | 1 | 1 | 1 | 1 | **1** |
| **Subcaso (b)** | 1 | 0 | 0 | 1 | **1** |
| | 1 | 0 | 1 | 2 | **1** |
| | 0 | 1 | 0 | -1 | **0** |
| | 0 | 1 | 1 | 0 | **0** |

O algoritmo baseia-se em como os seres humanos adquirem e compartilham conhecimento ao longo da vida. As dimensões de atualização afetadas por cada estágio evoluem dinamicamente conforme as iterações (avaliações) progridem:


$$d_{junior} = d \times \left( 1 - \frac{NFE}{MaxNFE} \right)^K$$

$$d_{senior} = d - d_{junior}$$


#### Fase Senior (mais detalhes em breve)
Agora na fase senior temos influências mais radicais , agora temos 3 partições (p) de individuos os melhores se  p =0.1 temos os melhores 10% os piores 10% e os 80% compondo os medianos.
se um elemento é pior 
$$xtk_new = xt + kf * [(x_{pbest} - x_{pworst}) + (x_{middle} - x_{t})]$$
ou se o elemento t possuir um desempenho pior que o elemento medio aleatorio .
$$xtk_new = xt + kf * [(xp_{best} - x_{pworst}) + (x_{middle} - x_{t})]$$


Pseudo codigo do artigo numero 2:
```pseudo
for t = 1 : Np do
    for k = 1 : d do
        if rand ≤ kR (knowledge ratio) then
            if f(xt) > f(xmiddle) then
                xtk_new = xt + kf * [(xpbest - xpworst) + (xmiddle - xt)]
            else
                xtk_new = xt + kf * [(xpbest - xpworst) + (xt - xmiddle)]
            end if
        else
            xtk_new = xtk_old
        end if
    end for
end for
```

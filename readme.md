# Rede Neural RBF

## 1. Ideia geral

A rede neural RBF, do inglês _Radial Basis Function_, é uma rede usada para problemas de classificação e regressão.

Ela é formada por três partes principais:

- Camada de entrada;
- Camada oculta com funções radiais;
- Camada de saída linear.

A principal ideia da RBF é medir a distância entre uma amostra de entrada e alguns pontos chamados **centros**.

Quanto mais próxima a amostra estiver de um centro, maior será a ativação do neurônio correspondente.

---

## 2. Estrutura da rede

A estrutura básica da rede RBF é:

```text
Entrada → Camada RBF → Saída
```

No problema utilizado, a entrada é formada pelos atributos da base de dados.

A saída representa uma das classes do problema.

Como o problema possui três classes, a saída pode ser representada no formato binário, também chamado de **one-hot**:

| Classe | Representação |
| :----: | :-----------: |
|   1    |  `[1; 0; 0]`  |
|   2    |  `[0; 1; 0]`  |
|   3    |  `[0; 0; 1]`  |

---

## 3. Função de ativação Gaussiana

A função usada na camada oculta é a função Gaussiana:

$$
\phi(x) = e^{-\frac{||x - c||^2}{2\sigma^2}}
$$

Onde:

- $x$ é a amostra de entrada;
- $c$ é o centro do neurônio;
- $||x - c||$ é a distância entre a entrada e o centro;
- $\sigma$ controla a largura da função Gaussiana.

Se a amostra estiver próxima do centro, a saída da função será próxima de 1.

Se a amostra estiver distante do centro, a saída será próxima de 0.

---

## 4. Matriz de ativações

Para cada amostra de treinamento, calcula-se a ativação em relação a cada centro.

Essas ativações formam a matriz `F`.

Também é adicionado um bias, formando a matriz `Fb`:

$$
F_b =
\begin{bmatrix}
-1 \\
F
\end{bmatrix}
$$

O bias ajuda a ajustar melhor a saída da rede.

---

## 5. Cálculo dos pesos da saída

A camada de saída combina linearmente as ativações da camada RBF.

A saída estimada é dada por:

$$
\hat{Y} = N F_b
$$

Onde:

- $\hat{Y}$ é a saída estimada;
- $N$ é a matriz de pesos da saída;
- $F_b$ é a matriz de ativações com bias.

Os pesos da saída podem ser calculados por mínimos quadrados ou pela pseudoinversa:

$$
N = Y F_b^{+}
$$

No código, isso pode ser feito com:

```octave
N = y_train * pinv(Fb);
```

---

## 6. Normalização dos dados

Como a RBF depende de distância, é importante normalizar os atributos.

Neste trabalho foi utilizada a normalização por **z-score**:

$$
z = \frac{x - \mu}{s}
$$

Onde:

- $x$ é o valor original;
- $\mu$ é a média do atributo;
- $s$ é o desvio padrão do atributo.

A média e o desvio padrão devem ser calculados usando apenas os dados de treino.

Depois, os mesmos valores são usados para normalizar os dados de teste.

Isso evita que informações do teste sejam usadas durante o treinamento.

---

## 7. Classificação

Após calcular a saída da rede, a classe escolhida é aquela com maior valor no vetor de saída.

$$
classe = \arg\max(\hat{Y})
$$

Exemplo:

```text
Saída da rede = [0.10; 0.85; 0.20]
Classe predita = 2
```

Nesse exemplo, a segunda posição possui o maior valor, então a classe predita é a classe 2.

---

## 8. Resultados

Resultados com Z-score aplicado:

| Métrica                 | Valor  |
| :---------------------- | :----  |
| Acurácia                | 82.58% |
| Número de amostras      |  310   |
| Número de atributos     |     6  |
| Número de classes       | 3      |
| Número de neurônios RBF |   34   |
| Valor de $\sigma$       |    1   |



Resultados sem Z-score aplicado:

| Métrica                 | Valor  |
| :---------------------- | :----  |
| Acurácia                | 48.39% |
| Número de amostras      |  310   |
| Número de atributos     |     6  |
| Número de classes       | 3      |
| Número de neurônios RBF |   34   |
| Valor de $\sigma$       |    1   |

### Observações

Os centróides foram iniciados de forma aleatória.

## 9. Conclusão

A rede RBF utiliza funções Gaussianas para medir a proximidade entre as amostras e os centros da rede.

Como o cálculo depende de distâncias, a normalização dos dados é uma etapa importante.

No problema com três classes, a saída é representada no formato one-hot, e a classe final é escolhida pela posição de maior valor na saída da rede.

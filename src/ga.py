import numpy as np
from random import sample

from benchmark import decode


class IndividuoGA:
    def __init__(self, numero_ops):
        self.cromossomo = 
        self.fitness =

    def __repr__(self):
        return f"IndividuoGA({self.cromossomo}, fitness={self.fitness:.2f})"



def selecao_torneio(pop, tamanho=3):
    # Seleção por torneio: seleciona aleatoriamente 'tamanho' indivíduos e retorna o melhor (menor fitness)
    torneio = sample(pop, tamanho)
    torneio.sort(key=lambda ind: ind.fitness)
    return torneio[0]


def crossover(ind1, ind2):
    # Realiza o crossover de um ponto entre dois indivíduos
    crom1 = ind1.cromossomo
    crom2 = ind2.cromossomo
    tamanho = len(crom1)
    ponto = np.random.randint(1, high=tamanho)
    novo_crom = crom1[:ponto] + crom2[ponto:]
    return IndividuoGA(novo_crom)


def mutacao(ind, taxa=0.05):
    # Aplica mutação trocando dois genes se ocorrer a chance definida pela taxa
    novo_crom = ind.cromossomo.copy()
    if np.random.rand() <= taxa:
        i, j = sample(range(len(novo_crom)), 2)
        if novo_crom[i] != novo_crom[j]:
            novo_crom[i], novo_crom[j] = novo_crom[j], novo_crom[i]
    return IndividuoGA(novo_crom)


def ga(jobs, seed, pop_size=100, iters=100):
    np.random.seed(seed)
    """
    Implementa o Algoritmo Genético (GA):
    - Inicializa uma população de soluções (cromossomos).
    - Avalia o fitness de cada indivíduo.
    - Executa iterações de seleção, crossover e mutação para gerar novas populações.
    - Retorna o melhor indivíduo (solução) encontrado.
    """

    numero_tarefas = sum(len(job) for job in jobs)
    ops = [j for j in range(len(jobs)) for _ in range(len(jobs[j]))]

    populacao =
    populacao = [
        IndividuoGA(gerar_individuo(numero_tarefas, len(vms))) for _ in range(pop_size)
    ]
    for ind in populacao:
        ind.fitness = calcula_fitness(ind.cromossomo, workflow, vms)
    for it in range(iters):
        nova_pop = []
        while len(nova_pop) < pop_size:
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            filho = crossover(pai1, pai2)
            filho = mutacao(filho)
            filho.fitness = calcula_fitness(filho.cromossomo, workflow, vms)
            nova_pop.append(filho)
        populacao = nova_pop
    melhor = min(populacao, key=lambda ind: ind.fitness)
    return melhor

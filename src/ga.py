import numpy as np
import random

from collections import Counter
from benchmark import decode, load_instance

class IndividuoGA:
    def __init__(self, lista_ops, jobs):
        self.cromossomo = lista_ops.copy()
        self.fitness = self.calcula_fitness(jobs)

    def __repr__(self):
        return f"IndividuoGA({self.cromossomo}, fitness={self.fitness:.2f})"
    
    def calcula_fitness(self, jobs):
        self.fitness = decode(jobs, self.cromossomo)
        return self.fitness

def selecao_torneio(pop, tamanho=3):
    # Seleção por torneio: seleciona aleatoriamente 'tamanho' indivíduos e retorna o melhor (menor fitness)
    torneio = random.sample(pop, tamanho)
    torneio.sort(key=lambda ind: ind.fitness)
    return torneio[0]

def order_crossover(p1: list, p2: list) -> list:
    """
    One-point Order Crossover (OX) adaptado para permutações com elementos repetidos:
    - Seleciona um índice de corte `cut` (>0, <len(p1)).
    - Copia p1[0:cut] para o filho.
    - Calcula o multiconjunto de genes faltantes (diferença de contagens entre p1 e prefixo copiado).
    - Preenche child[cut:] usando a ordem de p2, sem ultrapassar o número de ocorrências de cada gene.
    """
    D = len(p1)
    cut = np.random.randint(1, D)

    child = [None] * D
    child[:cut] = p1[:cut]

    cnt_p1 = Counter(p1)
    cnt_prefix = Counter(child[:cut])

    missing = []
    for gene, total in cnt_p1.items():
        needed = total - cnt_prefix.get(gene, 0)
        missing.extend([gene] * needed)

    pos = cut
    for gene in p2:
        if gene in missing:
            child[pos] = gene
            missing.remove(gene)
            pos += 1
            if pos >= D:
                break

    return child

def crossover(ind1, ind2, jobs):
    p1 = list(ind1.cromossomo)
    p2 = list(ind2.cromossomo)
    novo_crom = order_crossover(p1, p2)
    
    # opcional: verifique integridade
    assert sorted(novo_crom) == sorted(p1), "Cromossomo inválido!"
    
    filho = IndividuoGA(novo_crom, jobs)
    filho.calcula_fitness(jobs)
    return filho


def mutacao(ind, jobs, taxa=0.05):
    # Aplica mutação trocando dois genes se ocorrer a chance definida pela taxa
    novo_crom = ind.cromossomo.copy()
    if np.random.rand() <= taxa:
        i, j = random.sample(range(len(novo_crom)), 2)
        if novo_crom[i] != novo_crom[j]:
            novo_crom[i], novo_crom[j] = novo_crom[j], novo_crom[i]
    return IndividuoGA(novo_crom, jobs)


def ga(jobs, seed, pop_size=100, iters=100):
    np.random.seed(seed)
    """
    Implementa o Algoritmo Genético (GA):
    - Inicializa uma população de soluções (cromossomos).
    - Avalia o fitness de cada indivíduo.
    - Executa iterações de seleção, crossover e mutação para gerar novas populações.
    - Retorna o melhor indivíduo (solução) encontrado.
    """

    ops = [j for j in range(len(jobs)) for _ in range(len(jobs[j]))]
    populacao = [
        IndividuoGA(np.random.permutation(ops), jobs) for _ in range(pop_size)
    ]

    for ind in populacao:
        ind.calcula_fitness(jobs)
    for it in range(iters):
        nova_pop = []
        while len(nova_pop) < pop_size:
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            filho = crossover(pai1, pai2, jobs=jobs)
            filho = mutacao(filho, jobs=jobs)
            nova_pop.append(filho)
        populacao = nova_pop
    melhor = min(populacao, key=lambda ind: ind.fitness)
    
    return melhor.fitness
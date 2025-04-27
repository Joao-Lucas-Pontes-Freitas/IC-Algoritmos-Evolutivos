import numpy as np
import random

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
    Order Crossover (OX) para permutações:
    - Seleciona dois pontos de corte i<j.
    - Copia p1[i:j] para o filho.
    - Preenche o restante em ordem de p2, rodando em círculo.
    """
    D = len(p1)
    index = np.random.randint(0, len(p1))
    
    child = [] * D
    child[:index] = p1[:index]

    faltando = np.array([])

    unique, counts = np.unique(p1, return_counts=True)

    faltando = []
    for gene, total in zip(unique, counts):
        # quantas vezes o gene já apareceu em 'child' (ignorando None)
        presente = child.count(gene)
        # quantas cópias faltam
        missing = total - presente
        if missing > 0:
            # adiciona 'missing' cópias do gene à lista de faltantes
            faltando.extend([gene] * missing)

    # se precisar como numpy array:
    faltando = np.array(faltando, dtype=int)

    pos = index
    for gene in faltando:
        if pos >= D:
            pos = 0
        child[pos] = gene
        pos += 1
    
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
    return melhor

ga(load_instance("abz5"), 42)   



# benchmarks = ["abz5", "abz6", "abz7", "abz8", "abz9", "ft06", "ft10", "ft20", "yn1", "yn2", "yn3", "yn4"]
# print('-' * 30)
# print()

# for benchmark in benchmarks:
#     print(f"Executando benchmark: {benchmark}")
    
#     jobs = load_instance(benchmark)

#     seed = 42
#     experimentos = 1
#     resultados_ga = np.zeros(experimentos)

#     for i in range(experimentos):
#         ga_val = ga(jobs, seed + i)
#         resultados_ga[i] = ga_val.fitness

#     print("GA - Média: ", np.mean(resultados_ga))
#     print("GA - Melhor: ", np.min(resultados_ga))
#     print("GA - Pior: ", np.max(resultados_ga))
#     print('-' * 30)
#     print()
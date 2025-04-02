import random
import statistics
import time
import matplotlib.pyplot as plt

# Define seed fixo para reprodutibilidade
random.seed(42)

# Fator de escala para ajustar os tempos de execução
SCALE_FACTOR = 1

###############################
# Classes e Funções Básicas
###############################


class Tarefa:
    def __init__(self, id, carga):
        self.id = id
        self.carga = carga  # carga de trabalho em MI (Milhões de Instruções)
        self.dependencias = (
            []
        )  # (não utilizado aqui, pois usamos workflows sem dependências)

    def __repr__(self):
        return f"Tarefa(id={self.id}, carga={self.carga})"


class Workflow:
    def __init__(self, tarefas=None):
        self.tarefas = tarefas if tarefas is not None else []

    def __repr__(self):
        return f"Workflow({self.tarefas})"


class VM:
    def __init__(self, id, mips, custo):
        self.id = id
        self.mips = mips  # MI/s
        self.custo = custo  # custo por segundo
        self.tempo_acumulado = 0  # tempo total de execução das tarefas alocadas

    def reset(self):
        self.tempo_acumulado = 0

    def __repr__(self):
        return f"VM(id={self.id}, mips={self.mips}, custo={self.custo})"


def simular_workflow(individuo, workflow, vms):
    # Reseta os tempos acumulados de cada VM
    for vm in vms:
        vm.reset()
    custo_total = 0.0
    for idx, tarefa in enumerate(workflow.tarefas):
        vm_idx = individuo[idx]
        vm = vms[vm_idx]
        # Calcula o tempo de execução com o fator de escala
        tempo_execucao = (tarefa.carga / vm.mips) * SCALE_FACTOR
        vm.tempo_acumulado += tempo_execucao
        custo_total += tempo_execucao * vm.custo
    makespan = max(vm.tempo_acumulado for vm in vms)
    tempos = [vm.tempo_acumulado for vm in vms]
    balanceamento = statistics.stdev(tempos) if len(tempos) > 1 else 0
    return makespan, custo_total, balanceamento


###############################
# Implementação do GA
###############################


def gerar_individuo(num_tarefas, num_vms):
    return [random.randint(0, num_vms - 1) for _ in range(num_tarefas)]


class Individuo:
    def __init__(self, cromossomo):
        self.cromossomo = cromossomo
        self.fitness = None

    def __repr__(self):
        return f"Individuo({self.cromossomo}, fitness={self.fitness})"


def calcula_fitness(individuo, workflow, vms, alpha1=0.4, alpha2=0.4, alpha3=0.2):
    makespan, custo_total, balanceamento = simular_workflow(individuo, workflow, vms)
    return alpha1 * makespan + alpha2 * custo_total + alpha3 * balanceamento


def selecao_torneio(populacao, tamanho_torneio=3):
    torneio = random.sample(populacao, tamanho_torneio)
    torneio.sort(key=lambda ind: ind.fitness)
    return torneio[0]


def crossover(ind1, ind2):
    crom1 = ind1.cromossomo
    crom2 = ind2.cromossomo
    tamanho = len(crom1)
    ponto_corte = random.randint(1, tamanho - 1)
    novo_cromossomo = crom1[:ponto_corte] + crom2[ponto_corte:]
    return Individuo(novo_cromossomo)


def mutacao(ind, taxa_mutacao=0.05, num_vms=3):
    novo_crom = ind.cromossomo.copy()
    if random.random() <= taxa_mutacao:
        i, j = random.sample(range(len(novo_crom)), 2)
        # Se os genes forem diferentes, troca os valores
        if novo_crom[i] != novo_crom[j]:
            novo_crom[i], novo_crom[j] = novo_crom[j], novo_crom[i]
    return Individuo(novo_crom)


def ga(
    pop_inicial,
    workflow,
    vms,
    num_iteracoes,
    tamanho_torneio=3,
    taxa_mutacao=0.05,
    num_vms=3,
):
    populacao = pop_inicial
    for ind in populacao:
        ind.fitness = calcula_fitness(ind.cromossomo, workflow, vms)
    for it in range(num_iteracoes):
        nova_populacao = []
        while len(nova_populacao) < len(populacao):
            pai1 = selecao_torneio(populacao, tamanho_torneio)
            pai2 = selecao_torneio(populacao, tamanho_torneio)
            filho = crossover(pai1, pai2)
            filho = mutacao(filho, taxa_mutacao, num_vms)
            filho.fitness = calcula_fitness(filho.cromossomo, workflow, vms)
            nova_populacao.append(filho)
        populacao = nova_populacao
    return populacao


###############################
# Implementação do PSO
###############################


class Particula:
    def __init__(self, posicao):
        self.posicao = posicao  # alocação de tarefas (lista de inteiros)
        self.fitness = None
        self.pbest = posicao.copy()
        self.fitness_pbest = None
        self.velocity = [0.0] * len(posicao)

    def __repr__(self):
        return f"Particula(pos={self.posicao}, fitness={self.fitness})"


def atualizar_pbest(pop_particulas, workflow, vms):
    for particula in pop_particulas:
        particula.fitness = calcula_fitness(particula.posicao, workflow, vms)
        if (
            particula.fitness_pbest is None
            or particula.fitness < particula.fitness_pbest
        ):
            particula.pbest = particula.posicao.copy()
            particula.fitness_pbest = particula.fitness


def obter_gbest(pop_particulas):
    melhor = min(pop_particulas, key=lambda p: p.fitness_pbest)
    return melhor.pbest.copy(), melhor.fitness_pbest


def atualizar_velocidade(particula, gbest, C1=1.0, C2=1.1):
    # r1 e r2 são gerados uma única vez por iteração (para cada chamada individual de atualização, será a mesma para todos os genes)
    r1 = random.random()
    r2 = random.random()
    for i in range(len(particula.posicao)):
        if particula.posicao[i] == particula.pbest[i]:
            particula.velocity[i] -= C1 * r1
        else:
            particula.velocity[i] += C1 * r1
        if particula.posicao[i] == gbest[i]:
            particula.velocity[i] -= C2 * r2
        else:
            particula.velocity[i] += C2 * r2


def atualizar_posicao(particula):
    if len(particula.velocity) < 2:
        return
    max1 = max(range(len(particula.velocity)), key=lambda i: particula.velocity[i])
    indices = list(range(len(particula.velocity)))
    indices.remove(max1)
    max2 = max(indices, key=lambda i: particula.velocity[i])
    particula.posicao[max1], particula.posicao[max2] = (
        particula.posicao[max2],
        particula.posicao[max1],
    )


def pso(pop_particulas, workflow, vms, num_iteracoes, C1=1.0, C2=1.1):
    for it in range(num_iteracoes):
        atualizar_pbest(pop_particulas, workflow, vms)
        gbest, gbest_fitness = obter_gbest(pop_particulas)
        for particula in pop_particulas:
            atualizar_velocidade(particula, gbest, C1, C2)
        for particula in pop_particulas:
            atualizar_posicao(particula)
    return pop_particulas


###############################
# Integração Híbrida: GA-PSO
###############################


def ga_pso_hibrido(
    workflow, vms, num_tarefas, num_vms, pop_size=100, iteracoes_total=100
):
    iter_ga = iteracoes_total // 2
    iter_pso = iteracoes_total - iter_ga
    populacao = [
        Individuo(gerar_individuo(num_tarefas, num_vms)) for _ in range(pop_size)
    ]
    populacao = ga(
        populacao,
        workflow,
        vms,
        iter_ga,
        tamanho_torneio=3,
        taxa_mutacao=0.05,
        num_vms=num_vms,
    )
    pop_particulas = [Particula(ind.cromossomo.copy()) for ind in populacao]
    pop_particulas = pso(pop_particulas, workflow, vms, iter_pso, C1=1.0, C2=1.1)
    gbest, gbest_fitness = obter_gbest(pop_particulas)
    return gbest, gbest_fitness


###############################
# Funções para Experimentos
###############################


# Utiliza os benchmarks do artigo para carga das tarefas
def criar_workflow(num_tarefas, fixed_loads):
    tarefas = []
    if num_tarefas in fixed_loads:
        carga_valor = fixed_loads[num_tarefas]
        for i in range(num_tarefas):
            tarefas.append(Tarefa(i, carga=carga_valor))
    else:
        for i in range(num_tarefas):
            tarefas.append(Tarefa(i, carga=random.randint(10000, 100000)))
    return Workflow(tarefas)


# Para as VMs, usaremos valores médios fixos conforme benchmark
def criar_vms(num_vms):
    vms = []
    for i in range(num_vms):
        # MIPS médio = 875, custo médio = 0.085
        vms.append(VM(id=i, mips=875, custo=0.085))
    return vms


def executar_experimento(
    num_tarefas,
    num_vms,
    pop_size=100,
    iteracoes_total=100,
    num_execucoes=2,
    fixed_loads={},
):
    resultados = []
    for execucao in range(num_execucoes):
        wf = criar_workflow(num_tarefas, fixed_loads)
        vms = criar_vms(num_vms)
        melhor_solucao, fitness_melhor = ga_pso_hibrido(
            wf, vms, num_tarefas, num_vms, pop_size, iteracoes_total
        )
        makespan, custo_total, balanceamento = simular_workflow(melhor_solucao, wf, vms)
        resultados.append((makespan, custo_total, balanceamento))
    makespans = [r[0] for r in resultados]
    custos = [r[1] for r in resultados]
    balances = [r[2] for r in resultados]
    media_makespan = statistics.mean(makespans)
    media_custo = statistics.mean(custos)
    media_balance = statistics.mean(balances)
    return media_makespan, media_custo, media_balance


###############################
# Bloco Principal para Execução
###############################

if __name__ == "__main__":
    # Benchmarks fixos conforme o artigo:
    # Tabela 4 - Características dos workflows Montage:
    fixed_loads = {
        25: 53200,  # Cenário 1: 25 tarefas
        50: 32480,  # Cenário 2: 50 tarefas
        100: 32760,  # Cenário 3: 100 tarefas
        1000: 22203,  # Cenário 4: 1000 tarefas
    }
    # Número de VMs fixo conforme o artigo
    num_vms = 16

    # Cenários: 25, 50, 100 e 1000 tarefas
    cenarios = [25, 50, 100, 1000]
    resultados_experimentos = {}

    inicio = time.time()
    for num_tarefas in cenarios:
        media_makespan, media_custo, media_balance = executar_experimento(
            num_tarefas,
            num_vms,
            pop_size=100,
            iteracoes_total=100,
            num_execucoes=5,
            fixed_loads=fixed_loads,
        )
        resultados_experimentos[num_tarefas] = {
            "makespan": media_makespan,
            "custo": media_custo,
            "balanceamento": media_balance,
        }
        print(
            f"Cenário {num_tarefas} tarefas: Makespan = {media_makespan:.2f}, Custo = {media_custo:.2f}, Balanceamento = {media_balance:.2f}"
        )
    fim = time.time()
    print(f"Tempo total dos experimentos: {fim - inicio:.2f} segundos")
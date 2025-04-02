import random
import statistics
from models import Tarefa, Workflow, VM

# Fator de escala para ajustar os tempos (se necessário)
SCALE_FACTOR = 1

# --- Simulação do Workflow e Cálculo de Fitness ---
def simular_workflow(individuo, workflow, vms):
    for vm in vms:
        vm.reset()
    custo_total = 0.0
    for idx, tarefa in enumerate(workflow.tarefas):
        vm_idx = individuo[idx]
        vm = vms[vm_idx]
        tempo_execucao = (tarefa.carga / vm.mips) * SCALE_FACTOR
        vm.tempo_acumulado += tempo_execucao
        custo_total += tempo_execucao * vm.custo
    makespan = max(vm.tempo_acumulado for vm in vms)
    tempos = [vm.tempo_acumulado for vm in vms]
    balanceamento = statistics.stdev(tempos) if len(tempos) > 1 else 0
    return makespan, custo_total, balanceamento

def calcula_fitness(individuo, workflow, vms, alpha1=0.4, alpha2=0.4, alpha3=0.2):
    makespan, custo_total, balanceamento = simular_workflow(individuo, workflow, vms)
    return alpha1 * makespan + alpha2 * custo_total + alpha3 * balanceamento

# --- Algoritmo Genético (GA) ---
class Individuo:
    def __init__(self, cromossomo):
        self.cromossomo = cromossomo
        self.fitness = None

    def __repr__(self):
        return f"Individuo({self.cromossomo}, fitness={self.fitness})"

def gerar_individuo(num_tarefas, num_vms):
    return [random.randint(0, num_vms - 1) for _ in range(num_tarefas)]

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
        if novo_crom[i] != novo_crom[j]:
            novo_crom[i], novo_crom[j] = novo_crom[j], novo_crom[i]
    return Individuo(novo_crom)

def ga(pop_inicial, workflow, vms, num_iteracoes, tamanho_torneio=3, taxa_mutacao=0.05, num_vms=3):
    populacao = pop_inicial
    for ind in populacao:
        ind.fitness = calcula_fitness(ind.cromossomo, workflow, vms)
    for _ in range(num_iteracoes):
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

# --- Particle Swarm Optimization (PSO) ---
class Particula:
    def __init__(self, posicao):
        self.posicao = posicao  # Solução (alocação de tarefas)
        self.fitness = None
        self.pbest = posicao.copy()
        self.fitness_pbest = None
        self.velocity = [0.0] * len(posicao)

    def __repr__(self):
        return f"Particula(pos={self.posicao}, fitness={self.fitness})"

def atualizar_pbest(pop_particulas, workflow, vms):
    for particula in pop_particulas:
        particula.fitness = calcula_fitness(particula.posicao, workflow, vms)
        if particula.fitness_pbest is None or particula.fitness < particula.fitness_pbest:
            particula.pbest = particula.posicao.copy()
            particula.fitness_pbest = particula.fitness

def obter_gbest(pop_particulas):
    melhor = min(pop_particulas, key=lambda p: p.fitness_pbest)
    return melhor.pbest.copy(), melhor.fitness_pbest

def atualizar_velocidade(particula, gbest, C1=1.0, C2=1.1):
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
    particula.posicao[max1], particula.posicao[max2] = particula.posicao[max2], particula.posicao[max1]

def pso(pop_particulas, workflow, vms, num_iteracoes, C1=1.0, C2=1.1):
    for _ in range(num_iteracoes):
        atualizar_pbest(pop_particulas, workflow, vms)
        gbest, _ = obter_gbest(pop_particulas)
        for particula in pop_particulas:
            atualizar_velocidade(particula, gbest, C1, C2)
        for particula in pop_particulas:
            atualizar_posicao(particula)
    return pop_particulas

# --- Algoritmo Híbrido GA-PSO ---
def ga_pso_hibrido(workflow, vms, num_tarefas, num_vms, pop_size=100, iteracoes_total=100):
    iter_ga = iteracoes_total // 2
    iter_pso = iteracoes_total - iter_ga
    populacao = [Individuo(gerar_individuo(num_tarefas, num_vms)) for _ in range(pop_size)]
    populacao = ga(populacao, workflow, vms, iter_ga, tamanho_torneio=3, taxa_mutacao=0.05, num_vms=num_vms)
    pop_particulas = [Particula(ind.cromossomo.copy()) for ind in populacao]
    pop_particulas = pso(pop_particulas, workflow, vms, iter_pso, C1=1.0, C2=1.1)
    gbest, gbest_fitness = obter_gbest(pop_particulas)
    return gbest, gbest_fitness

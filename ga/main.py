import time
import statistics
from benchmarks import gerar_cenarios_benchmark
from algorithms import (
    ga, pso, ga_pso_hibrido, gerar_individuo,
    Particula, obter_gbest, simular_workflow, calcula_fitness, Individuo
)
from models import VM

def executar_experimento_ga(workflow, vms, pop_size, iteracoes_total, num_execucoes):
    resultados = []
    tempos_execucao = []
    num_tarefas = len(workflow.tarefas)
    num_vms = len(vms)
    for _ in range(num_execucoes):
        # Cria cópias das VMs para evitar interferência entre execuções
        vms_exec = [VM(vm.id, vm.mips, vm.custo) for vm in vms]
        populacao = [Individuo(gerar_individuo(num_tarefas, num_vms)) for _ in range(pop_size)]
        inicio = time.time()
        populacao = ga(populacao, workflow, vms_exec, iteracoes_total, tamanho_torneio=3, taxa_mutacao=0.05, num_vms=num_vms)
        fim = time.time()
        tempos_execucao.append(fim - inicio)
        melhor_ind = min(populacao, key=lambda ind: ind.fitness)
        makespan, custo_total, balanceamento = simular_workflow(melhor_ind.cromossomo, workflow, vms_exec)
        fitness = calcula_fitness(melhor_ind.cromossomo, workflow, vms_exec)
        resultados.append((makespan, custo_total, balanceamento, fitness))
    media_resultados = [statistics.mean([r[i] for r in resultados]) for i in range(4)]
    media_tempo = statistics.mean(tempos_execucao)
    return media_resultados, media_tempo

def executar_experimento_pso(workflow, vms, pop_size, iteracoes_total, num_execucoes):
    resultados = []
    tempos_execucao = []
    num_tarefas = len(workflow.tarefas)
    num_vms = len(vms)
    for _ in range(num_execucoes):
        vms_exec = [VM(vm.id, vm.mips, vm.custo) for vm in vms]
        pop_particulas = [Particula(gerar_individuo(num_tarefas, num_vms)) for _ in range(pop_size)]
        inicio = time.time()
        pop_particulas = pso(pop_particulas, workflow, vms_exec, iteracoes_total, C1=1.0, C2=1.1)
        fim = time.time()
        tempos_execucao.append(fim - inicio)
        gbest, _ = obter_gbest(pop_particulas)
        makespan, custo_total, balanceamento = simular_workflow(gbest, workflow, vms_exec)
        fitness = calcula_fitness(gbest, workflow, vms_exec)
        resultados.append((makespan, custo_total, balanceamento, fitness))
    media_resultados = [statistics.mean([r[i] for r in resultados]) for i in range(4)]
    media_tempo = statistics.mean(tempos_execucao)
    return media_resultados, media_tempo

def executar_experimento_hibrido(workflow, vms, pop_size, iteracoes_total, num_execucoes):
    resultados = []
    tempos_execucao = []
    num_tarefas = len(workflow.tarefas)
    num_vms = len(vms)
    for _ in range(num_execucoes):
        vms_exec = [VM(vm.id, vm.mips, vm.custo) for vm in vms]
        inicio = time.time()
        melhor_solucao, _ = ga_pso_hibrido(workflow, vms_exec, num_tarefas, num_vms, pop_size, iteracoes_total)
        fim = time.time()
        tempos_execucao.append(fim - inicio)
        makespan, custo_total, balanceamento = simular_workflow(melhor_solucao, workflow, vms_exec)
        fitness = calcula_fitness(melhor_solucao, workflow, vms_exec)
        resultados.append((makespan, custo_total, balanceamento, fitness))
    media_resultados = [statistics.mean([r[i] for r in resultados]) for i in range(4)]
    media_tempo = statistics.mean(tempos_execucao)
    return media_resultados, media_tempo

def main():
    num_execucoes = 2       # Número de execuções por cenário
    pop_size = 100          # Tamanho da população
    iteracoes_total = 100   # Iterações para cada algoritmo

    benchmarks = gerar_cenarios_benchmark()

    print("Iniciando experimentos comparativos:")
    print("Comparação entre GA puro, PSO puro e GA-PSO híbrido")
    
    for nome, (workflow, vms) in benchmarks.items():
        num_tarefas = len(workflow.tarefas)
        print(f"\nCenário com {num_tarefas} tarefas:")
        
        resultados_ga, tempo_ga = executar_experimento_ga(workflow, vms, pop_size, iteracoes_total, num_execucoes)
        resultados_pso, tempo_pso = executar_experimento_pso(workflow, vms, pop_size, iteracoes_total, num_execucoes)
        resultados_hibrido, tempo_hibrido = executar_experimento_hibrido(workflow, vms, pop_size, iteracoes_total, num_execucoes)
        
        print(f"  GA puro: Makespan = {resultados_ga[0]:.2f}, Custo = {resultados_ga[1]:.2f}, Balanceamento = {resultados_ga[2]:.2f}, Fitness = {resultados_ga[3]:.2f}, Tempo médio = {tempo_ga:.4f} s")
        print(f"  PSO puro: Makespan = {resultados_pso[0]:.2f}, Custo = {resultados_pso[1]:.2f}, Balanceamento = {resultados_pso[2]:.2f}, Fitness = {resultados_pso[3]:.2f}, Tempo médio = {tempo_pso:.4f} s")
        print(f"  GA-PSO híbrido: Makespan = {resultados_hibrido[0]:.2f}, Custo = {resultados_hibrido[1]:.2f}, Balanceamento = {resultados_hibrido[2]:.2f}, Fitness = {resultados_hibrido[3]:.2f}, Tempo médio = {tempo_hibrido:.4f} s")

if __name__ == "__main__":
    main()
import time
import statistics
from gerar_benchmarks import gerar_cenarios_benchmark
from algoritmos import (
    ga,
    pso,
    ga_pso_hibrido,
    gerar_individuo,
    Particula,
    obter_gbest,
    simular_workflow,
    VM,
    Individuo,
    calcula_fitness  # Função que calcula o fitness
)

def executar_experimento_ga(workflow, vms, pop_size, iteracoes_total, num_execucoes):
    resultados = []
    num_tarefas = len(workflow.tarefas)
    num_vms = len(vms)
    for _ in range(num_execucoes):
        # Cria novas VMs para cada execução para evitar acúmulo de tempo
        vms_exec = [VM(vm.id, vm.mips, vm.custo) for vm in vms]
        # Gera população inicial para GA
        populacao = [Individuo(gerar_individuo(num_tarefas, num_vms)) for _ in range(pop_size)]
        # Executa GA puro por iteracoes_total iterações
        populacao = ga(populacao, workflow, vms_exec, iteracoes_total, tamanho_torneio=3, taxa_mutacao=0.05, num_vms=num_vms)
        melhor_ind = min(populacao, key=lambda ind: ind.fitness)
        makespan, custo_total, balanceamento = simular_workflow(melhor_ind.cromossomo, workflow, vms_exec)
        fitness = calcula_fitness(melhor_ind.cromossomo, workflow, vms_exec)
        resultados.append((makespan, custo_total, balanceamento, fitness))
    media_makespan = statistics.mean([r[0] for r in resultados])
    media_custo = statistics.mean([r[1] for r in resultados])
    media_balance = statistics.mean([r[2] for r in resultados])
    media_fitness = statistics.mean([r[3] for r in resultados])
    return media_makespan, media_custo, media_balance, media_fitness

def executar_experimento_pso(workflow, vms, pop_size, iteracoes_total, num_execucoes):
    resultados = []
    num_tarefas = len(workflow.tarefas)
    num_vms = len(vms)
    for _ in range(num_execucoes):
        vms_exec = [VM(vm.id, vm.mips, vm.custo) for vm in vms]
        # Gera população inicial para PSO com indivíduos aleatórios
        pop_particulas = [Particula(gerar_individuo(num_tarefas, num_vms)) for _ in range(pop_size)]
        # Executa PSO puro por iteracoes_total iterações
        pop_particulas = pso(pop_particulas, workflow, vms_exec, iteracoes_total, C1=1.0, C2=1.1)
        gbest, _ = obter_gbest(pop_particulas)
        makespan, custo_total, balanceamento = simular_workflow(gbest, workflow, vms_exec)
        fitness = calcula_fitness(gbest, workflow, vms_exec)
        resultados.append((makespan, custo_total, balanceamento, fitness))
    media_makespan = statistics.mean([r[0] for r in resultados])
    media_custo = statistics.mean([r[1] for r in resultados])
    media_balance = statistics.mean([r[2] for r in resultados])
    media_fitness = statistics.mean([r[3] for r in resultados])
    return media_makespan, media_custo, media_balance, media_fitness

def executar_experimento_hibrido(workflow, vms, pop_size, iteracoes_total, num_execucoes):
    resultados = []
    num_tarefas = len(workflow.tarefas)
    num_vms = len(vms)
    for _ in range(num_execucoes):
        vms_exec = [VM(vm.id, vm.mips, vm.custo) for vm in vms]
        # Executa o algoritmo híbrido GA-PSO (metade iterações GA e metade PSO)
        melhor_solucao, _ = ga_pso_hibrido(workflow, vms_exec, num_tarefas, num_vms, pop_size, iteracoes_total)
        makespan, custo_total, balanceamento = simular_workflow(melhor_solucao, workflow, vms_exec)
        fitness = calcula_fitness(melhor_solucao, workflow, vms_exec)
        resultados.append((makespan, custo_total, balanceamento, fitness))
    media_makespan = statistics.mean([r[0] for r in resultados])
    media_custo = statistics.mean([r[1] for r in resultados])
    media_balance = statistics.mean([r[2] for r in resultados])
    media_fitness = statistics.mean([r[3] for r in resultados])
    return media_makespan, media_custo, media_balance, media_fitness

def main():
    # Parâmetros fixos definidos diretamente no código:
    num_execucoes = 2        # Número de execuções para cada cenário
    pop_size = 100           # Tamanho da população
    iteracoes_total = 100    # Número total de iterações para cada algoritmo

    # Gera os cenários de benchmark fixos (reprodutíveis) do arquivo gerar_benchmarks.py
    benchmarks = gerar_cenarios_benchmark()

    print("Iniciando experimentos comparativos:")
    print("Comparação entre GA puro, PSO puro e GA-PSO híbrido")
    inicio = time.time()

    for nome, (workflow, vms) in benchmarks.items():
        num_tarefas = len(workflow.tarefas)
        print(f"\nCenário com {num_tarefas} tarefas:")

        media_ga = executar_experimento_ga(workflow, vms, pop_size, iteracoes_total, num_execucoes)
        media_pso = executar_experimento_pso(workflow, vms, pop_size, iteracoes_total, num_execucoes)
        media_hibrido = executar_experimento_hibrido(workflow, vms, pop_size, iteracoes_total, num_execucoes)

        print(f"  GA puro: Makespan = {media_ga[0]:.2f}, Custo = {media_ga[1]:.2f}, Balanceamento = {media_ga[2]:.2f}, Fitness = {media_ga[3]:.2f}")
        print(f"  PSO puro: Makespan = {media_pso[0]:.2f}, Custo = {media_pso[1]:.2f}, Balanceamento = {media_pso[2]:.2f}, Fitness = {media_pso[3]:.2f}")
        print(f"  GA-PSO híbrido: Makespan = {media_hibrido[0]:.2f}, Custo = {media_hibrido[1]:.2f}, Balanceamento = {media_hibrido[2]:.2f}, Fitness = {media_hibrido[3]:.2f}")

    fim = time.time()
    print(f"\nTempo total dos experimentos: {fim - inicio:.2f} segundos")

if __name__ == "__main__":
    main()
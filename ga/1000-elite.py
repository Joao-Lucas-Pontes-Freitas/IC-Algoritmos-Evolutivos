import random
import statistics
import time
from collections import deque, defaultdict

# Parâmetros globais fixos (do artigo)
SCALE_FACTOR = 0.0005       # Fator de escala para o tempo de execução (ajustável)
ALPHA1 = 0.4                # Peso para makespan
ALPHA2 = 0.4                # Peso para custo
ALPHA3 = 0.2                # Peso para balanceamento

###############################
# Definição das Classes
###############################

class Tarefa:
    def __init__(self, id, carga, dependencias=None):
        """
        Representa uma tarefa do workflow.
        id: identificador único da tarefa.
        carga: carga de trabalho em MI (entre 1.200.000 e 7.200.000 MI).
        dependencias: lista de IDs de tarefas que devem ser concluídas antes desta.
        """
        self.id = id
        self.carga = carga
        self.dependencias = dependencias if dependencias is not None else []
    
    def __repr__(self):
        return f"Tarefa(id={self.id}, carga={self.carga:.0f}, deps={self.dependencias})"

class Workflow:
    def __init__(self, tarefas=None):
        """
        Representa um workflow composto por diversas tarefas.
        """
        self.tarefas = tarefas if tarefas is not None else []
    
    def __repr__(self):
        return f"Workflow({self.tarefas})"

class VM:
    def __init__(self, id, mips, custo, ram, bw, policy="TIME_SHARED"):
        """
        Representa uma máquina virtual (VM) com os seguintes atributos:
        id: identificador da VM.
        mips: capacidade de processamento (entre 250 e 1500 MI/s).
        custo: custo por segundo (fixo 0.28).
        ram: memória em MB (entre 256 e 1024).
        bw: banda em Mbps (entre 250 e 1500).
        policy: política de escalonamento (fixa em "TIME_SHARED").
        """
        self.id = id
        self.mips = mips
        self.custo = custo
        self.ram = ram
        self.bw = bw
        self.policy = policy
        self.tempo_acumulado = 0.0   # Momento em que a VM estará livre
    
    def reset(self):
        # Reseta o tempo acumulado para simulações futuras
        self.tempo_acumulado = 0.0
    
    def __repr__(self):
        return f"VM(id={self.id}, mips={self.mips}, custo={self.custo})"

###############################
# Função de Gerar VMs (valores aleatórios conforme Tabela 2)
###############################

def gerar_vms(n=16):
    # Gera n VMs com mips, ram e bw aleatórios dentro dos intervalos definidos
    vms = []
    for i in range(n):
        mips = random.randint(250, 1500)
        ram = random.randint(256, 1024)
        bw = random.randint(250, 1500)
        vms.append(VM(id=i, mips=mips, custo=0.28, ram=ram, bw=bw, policy="TIME_SHARED"))
    return vms

###############################
# Ordenação Topológica (Kahn)
###############################

def topological_sort(workflow):
    """
    Realiza a ordenação topológica das tarefas usando o algoritmo de Kahn.
    Isso garante que as tarefas sejam executadas respeitando as dependências.
    """
    tarefas_dict = {tarefa.id: tarefa for tarefa in workflow.tarefas}
    in_degree = {tarefa.id: 0 for tarefa in workflow.tarefas}
    grafo = defaultdict(list)
    for tarefa in workflow.tarefas:
        for dep in tarefa.dependencias:
            in_degree[tarefa.id] += 1
            grafo[dep].append(tarefa.id)
    fila = deque([tid for tid, deg in in_degree.items() if deg == 0])
    ordem = []
    while fila:
        current = fila.popleft()
        ordem.append(tarefas_dict[current])
        for suc in grafo[current]:
            in_degree[suc] -= 1
            if in_degree[suc] == 0:
                fila.append(suc)
    if len(ordem) != len(workflow.tarefas):
        raise ValueError("O workflow possui ciclos!")
    return ordem

###############################
# Simulação do Workflow
###############################

def simular_workflow(individuo, workflow, vms):
    """
    Simula a execução do workflow de acordo com a ordem topológica.
    Para cada tarefa, determina o VM atribuído e calcula:
    - Tempo de início considerando as dependências.
    - Tempo de execução com base na carga e na capacidade do VM.
    - Custo total e makespan (tempo total de execução).
    Também calcula o balanceamento com base no desvio padrão dos tempos das VMs.
    """
    for vm in vms:
        vm.reset()
    termino_tarefas = {}
    ordem = topological_sort(workflow)
    custo_total = 0.0
    for tarefa in ordem:
        vm_index = individuo[tarefa.id]
        vm = vms[vm_index]
        inicio_por_deps = 0.0
        if tarefa.dependencias:
            inicio_por_deps = max(termino_tarefas[dep] for dep in tarefa.dependencias)
        inicio = max(vm.tempo_acumulado, inicio_por_deps)
        tempo_execucao = (tarefa.carga / vm.mips) * SCALE_FACTOR
        termino = inicio + tempo_execucao
        termino_tarefas[tarefa.id] = termino
        vm.tempo_acumulado = termino
        custo_total += tempo_execucao * vm.custo
    makespan = max(termino_tarefas.values())
    tempos_vm = [vm.tempo_acumulado for vm in vms]
    balanceamento = statistics.stdev(tempos_vm) if len(tempos_vm) > 1 else 0.0
    return makespan, custo_total, balanceamento

def calcula_fitness(individuo, workflow, vms):
    """
    Calcula o fitness de um indivíduo (solução) combinando makespan, custo total e balanceamento.
    """
    makespan, custo_total, balanceamento = simular_workflow(individuo, workflow, vms)
    return ALPHA1 * makespan + ALPHA2 * custo_total + ALPHA3 * balanceamento

###############################
# Implementação do GA
###############################

class IndividuoGA:
    def __init__(self, cromossomo):
        self.cromossomo = cromossomo
        self.fitness = None
    
    def __repr__(self):
        return f"IndividuoGA({self.cromossomo}, fitness={self.fitness:.2f})"

def gerar_individuo(num_tarefas, num_vms):
    # Gera um cromossomo onde cada posição indica o índice da VM atribuída à tarefa
    return [random.randint(0, num_vms - 1) for _ in range(num_tarefas)]

def selecao_torneio(pop, tamanho=3):
    # Seleção por torneio: seleciona aleatoriamente 'tamanho' indivíduos e retorna o melhor (menor fitness)
    torneio = random.sample(pop, tamanho)
    torneio.sort(key=lambda ind: ind.fitness)
    return torneio[0]

def crossover(ind1, ind2):
    # Realiza o crossover de um ponto entre dois indivíduos
    crom1 = ind1.cromossomo
    crom2 = ind2.cromossomo
    tamanho = len(crom1)
    ponto = random.randint(1, tamanho - 1)
    novo_crom = crom1[:ponto] + crom2[ponto:]
    return IndividuoGA(novo_crom)

def mutacao(ind, taxa=0.05):
    # Aplica mutação trocando dois genes se ocorrer a chance definida pela taxa
    novo_crom = ind.cromossomo.copy()
    if random.random() <= taxa:
        i, j = random.sample(range(len(novo_crom)), 2)
        if novo_crom[i] != novo_crom[j]:
            novo_crom[i], novo_crom[j] = novo_crom[j], novo_crom[i]
    return IndividuoGA(novo_crom)

def ga(workflow, vms, pop_size=100, iteracoes=100):
    num_tarefas = len(workflow.tarefas)
    # Inicializa a população
    populacao = [IndividuoGA(gerar_individuo(num_tarefas, len(vms))) for _ in range(pop_size)]
    for ind in populacao:
        ind.fitness = calcula_fitness(ind.cromossomo, workflow, vms)
    
    for it in range(iteracoes):
        nova_pop = []
        # Implementa o elitismo: preserva os 5 melhores da população atual
        populacao.sort(key=lambda ind: ind.fitness)
        elite = populacao[:5]  # os 5 melhores
        
        # Gera os novos indivíduos para completar a população (pop_size - len(elite))
        while len(nova_pop) < (pop_size - len(elite)):
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            filho = crossover(pai1, pai2)
            filho = mutacao(filho)
            filho.fitness = calcula_fitness(filho.cromossomo, workflow, vms)
            nova_pop.append(filho)
        
        # Adiciona os indivíduos elitistas à nova população
        populacao = nova_pop + elite
    
    melhor = min(populacao, key=lambda ind: ind.fitness)
    return melhor


###############################
# Construindo o Workflow Montage com n tarefas
###############################

def gerar_dependencias(num_tarefas, p=0.5, seed=None):
    """
    Gera as dependências para cada tarefa:
    Para cada tarefa i (i > 0), para cada tarefa j com j < i, há uma chance p de j ser uma dependência de i.
    """
    if seed is not None:
        random.seed(seed)
    dependencias = {}
    for i in range(num_tarefas):
        if i == 0:
            dependencias[i] = []
        else:
            deps = [j for j in range(i) if random.random() < p]
            dependencias[i] = deps
    return dependencias

def construir_workflow(num_tarefas):
    # Constrói o workflow com 'num_tarefas', definindo cargas linearmente de 1.200.000 a 7.200.000 MI
    dependencias = gerar_dependencias(num_tarefas, p=0.5)
    tarefas = []
    carga_min = 12 * 100000  # 1.200.000 MI
    carga_max = 72 * 100000  # 7.200.000 MI
    for i in range(num_tarefas):
        carga = carga_min + (carga_max - carga_min) * i / (num_tarefas - 1)
        tarefas.append(Tarefa(id=i, carga=carga, dependencias=dependencias[i]))
    return Workflow(tarefas)

###############################
# Execução dos Experimentos (n execuções por cenário)
###############################

def experimentos(num_experimentos=500):
    """
    Para cada tamanho de workflow (25, 50, 100 tarefas), o algoritmo GA é executado num número definido de vezes.
    São calculados os resultados médios:
      - Fitness
      - Makespan
      - Custo Total
      - Balanceamento (desvio padrão dos tempos das VMs)
      - Tempo médio de execução de cada execução
    Além disso, é mostrado o tempo total gasto para todas as execuções do cenário.
    """
    random.seed(42)  # Para reprodutibilidade
    workflow_sizes = [500]
    num_vms = 16

    for n in workflow_sizes:
        print(f"\nExperimentos para workflow com {n} tarefas:")
        workflow = construir_workflow(n)
        vms = gerar_vms(n=num_vms)
        
        soma_tempo = 0.0
        soma_makespan = 0.0
        soma_custo = 0.0
        soma_balanceamento = 0.0
        soma_fitness = 0.0
        
        # Marca o tempo de início total para o cenário
        inicio_total = time.time()
        for i in range(num_experimentos):
            inicio = time.time()
            melhor_ind = ga(workflow, vms, pop_size=100, iteracoes=100)
            makespan, custo_total, balanceamento = simular_workflow(melhor_ind.cromossomo, workflow, vms)
            fitness = calcula_fitness(melhor_ind.cromossomo, workflow, vms)
            fim = time.time()
            
            soma_tempo += (fim - inicio)
            soma_makespan += makespan
            soma_custo += custo_total
            soma_balanceamento += balanceamento
            soma_fitness += fitness
        fim_total = time.time()
        tempo_total = fim_total - inicio_total
        
        print(f"Resultados médios após {num_experimentos} execuções:")
        print(f"Fitness médio: {soma_fitness / num_experimentos:.2f}")
        print(f"Makespan médio: {soma_makespan / num_experimentos:.2f} s")
        print(f"Custo Total médio: ${soma_custo / num_experimentos:.2f}")
        print(f"Balanceamento médio: {soma_balanceamento / num_experimentos:.2f}")
        print(f"Tempo médio de execução: {soma_tempo / num_experimentos:.4f} s")
        print(f"Tempo total para o cenário: {tempo_total:.4f} s")

if __name__ == "__main__":
    experimentos()
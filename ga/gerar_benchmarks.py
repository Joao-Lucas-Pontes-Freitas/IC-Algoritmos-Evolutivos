import random

# Classes básicas (ajuste conforme necessário se já as tiver definidas)
class Tarefa:
    def __init__(self, id, carga):
        self.id = id
        self.carga = carga  # carga de trabalho em MI (Milhões de Instruções)
        self.dependencias = []  # caso não haja dependências, manter vazio

    def __repr__(self):
        return f"Tarefa(id={self.id}, carga={self.carga})"


class Workflow:
    def __init__(self, tarefas=None):
        self.tarefas = tarefas if tarefas is not None else []

    def __repr__(self):
        return f"Workflow({self.tarefas})"


class VM:
    def __init__(
        self,
        id,
        mips,
        custo,
        ram,
        bw,
        processor_speed=10000,
        num_processors=4,
        policy="TIME_SHARED",
    ):
        self.id = id
        self.mips = mips
        self.custo = custo  # custo por segundo (ajuste conforme necessário)
        self.ram = ram
        self.bw = bw
        self.processor_speed = processor_speed
        self.num_processors = num_processors
        self.policy = policy
        self.tempo_acumulado = 0

    def reset(self):
        self.tempo_acumulado = 0

    def __repr__(self):
        return (
            f"VM(id={self.id}, mips={self.mips}, custo={self.custo}, "
            f"ram={self.ram}, bw={self.bw}, policy={self.policy})"
        )

def gerar_vms(n=16, seed=42):
    """
    Gera n VMs com características baseadas na Tabela 2:
      - MIPS entre 250 e 1500
      - RAM entre 256 e 1024 MB
      - BW entre 250 e 1500 mbps
      - Processor speed = 10000
      - num_processors = 4
      - policy = TIME_SHARED
    """
    random.seed(seed)  # para manter sempre os mesmos valores
    vms = []
    for i in range(n):
        mips = random.randint(250, 1500)
        ram = random.randint(256, 1024)
        bw = random.randint(250, 1500)
        # Define um custo fixo ou aleatório, caso queira variar
        custo = round(random.uniform(0.01, 0.05), 3)
        vms.append(
            VM(
                id=i,
                mips=mips,
                custo=custo,
                ram=ram,
                bw=bw,
                processor_speed=10000,
                num_processors=4,
                policy="TIME_SHARED",
            )
        )
    return vms

def gerar_tarefas(qtd_tarefas, seed=100):
    """
    Gera 'qtd_tarefas' Tarefas com cargas (MI) aleatórias.
    Você pode ajustar o intervalo de cargas conforme sua necessidade.
    """
    random.seed(seed)  # garante repetibilidade
    tarefas = []
    for i in range(qtd_tarefas):
        # Intervalo de carga pode ser ajustado livremente
        carga = random.randint(1_000, 100_000)
        tarefas.append(Tarefa(i, carga))
    return tarefas

def gerar_cenario(qtd_tarefas, vm_seed=42, task_seed=100):
    """
    Gera um cenário completo (Workflow + lista de VMs)
    com base na Tabela 2.
    """
    vms = gerar_vms(n=16, seed=vm_seed)
    tarefas = gerar_tarefas(qtd_tarefas, seed=task_seed)
    workflow = Workflow(tarefas)
    return workflow, vms

def gerar_cenarios_benchmark():
    """
    Gera os 4 cenários de benchmark, seguindo a Tabela 2:
      1) 25 tarefas
      2) 50 tarefas
      3) 100 tarefas
      4) 1000 tarefas
    Todos usam 16 VMs e as mesmas regras de formatação.
    Retorna um dicionário com cada cenário.
    """
    cenarios = {}
    # Cenário 1: 25 tarefas
    cenarios["cenario_25"] = gerar_cenario(qtd_tarefas=25, vm_seed=42, task_seed=100)
    # Cenário 2: 50 tarefas
    cenarios["cenario_50"] = gerar_cenario(qtd_tarefas=50, vm_seed=42, task_seed=200)
    # Cenário 3: 100 tarefas
    cenarios["cenario_100"] = gerar_cenario(qtd_tarefas=100, vm_seed=42, task_seed=300)
    # Cenário 4: 1000 tarefas
    cenarios["cenario_1000"] = gerar_cenario(qtd_tarefas=1000, vm_seed=42, task_seed=400)

    return cenarios

if __name__ == "__main__":
    # Exemplo de uso:
    todos_cenarios = gerar_cenarios_benchmark()
    for nome, (workflow, vms) in todos_cenarios.items():
        print(f"=== {nome.upper()} ===")
        print(f"Número de tarefas: {len(workflow.tarefas)}")
        print(f"Número de VMs: {len(vms)}")
        # Mostra as 5 primeiras tarefas e 2 primeiras VMs para exemplo
        print("Tarefas (exemplo):", workflow.tarefas[:5])
        print("VMs (exemplo):", vms[:2])
        print("-" * 40)

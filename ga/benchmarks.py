import random
from models import Tarefa, Workflow, VM

def gerar_vms(n=16, seed=42):
    random.seed(seed)
    vms = []
    for i in range(n):
        mips = random.randint(250, 1500)
        ram = random.randint(256, 1024)
        bw = random.randint(250, 1500)
        custo = round(random.uniform(0.01, 0.05), 3)
        vms.append(VM(id=i, mips=mips, custo=custo, ram=ram, bw=bw, policy="TIME_SHARED"))
    return vms

def gerar_tarefas(qtd_tarefas, seed=100):
    random.seed(seed)
    tarefas = []
    for i in range(qtd_tarefas):
        carga = random.randint(1_000, 100_000)
        tarefas.append(Tarefa(i, carga))
    return tarefas

def gerar_cenario(qtd_tarefas, vm_seed=42, task_seed=100):
    vms = gerar_vms(n=16, seed=vm_seed)
    tarefas = gerar_tarefas(qtd_tarefas, seed=task_seed)
    workflow = Workflow(tarefas)
    return workflow, vms

def gerar_cenarios_benchmark():
    cenarios = {}
    cenarios["cenario_25"] = gerar_cenario(qtd_tarefas=25, vm_seed=42, task_seed=100)
    cenarios["cenario_50"] = gerar_cenario(qtd_tarefas=50, vm_seed=42, task_seed=200)
    cenarios["cenario_100"] = gerar_cenario(qtd_tarefas=100, vm_seed=42, task_seed=300)
    cenarios["cenario_1000"] = gerar_cenario(qtd_tarefas=1000, vm_seed=42, task_seed=400)
    return cenarios

if __name__ == "__main__":
    cenarios = gerar_cenarios_benchmark()
    for nome, (workflow, vms) in cenarios.items():
        print(f"=== {nome.upper()} ===")
        print(f"Número de tarefas: {len(workflow.tarefas)}")
        print(f"Número de VMs: {len(vms)}")
        print("Tarefas (exemplo):", workflow.tarefas[:5])
        print("VMs (exemplo):", vms[:2])
        print("-" * 40)

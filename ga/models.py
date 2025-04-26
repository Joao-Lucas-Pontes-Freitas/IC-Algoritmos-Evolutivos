class Tarefa:
    def __init__(self, id, carga):
        self.id = id
        self.carga = carga  # carga de trabalho em MI (Milhões de Instruções)
        self.dependencias = []

    def __repr__(self):
        return f"Tarefa(id={self.id}, carga={self.carga})"


class Workflow:
    def __init__(self, tarefas=None):
        self.tarefas = tarefas if tarefas is not None else []

    def __repr__(self):
        return f"Workflow({self.tarefas})"


class VM:
    def __init__(self, id, mips, custo, ram=None, bw=None, policy="TIME_SHARED"):
        self.id = id
        self.mips = mips  # capacidade de processamento
        self.custo = custo  # custo por segundo
        self.ram = ram
        self.bw = bw
        self.policy = policy
        self.tempo_acumulado = 0

    def reset(self):
        self.tempo_acumulado = 0

    def __repr__(self):
        return f"VM(id={self.id}, mips={self.mips}, custo={self.custo})"

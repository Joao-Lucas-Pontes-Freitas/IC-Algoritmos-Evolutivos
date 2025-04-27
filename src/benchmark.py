from job_shop_lib.benchmarking import load_benchmark_instance


def load_instance(name):
    """
    Carrega um benchmark JSSP do job-shop-lib e retorna:
      List[List[(machine_id:int, duration:int)]]
    Aceita internamente instâncias com:
      - inst.durations_matrix + inst.machines_matrix
      - inst.processing_times
      - inst.jobs (tuplas ou objetos com .machine/.time ou .machine_id/.duration)
    """
    inst = load_benchmark_instance(name)

    # 1) durations_matrix + machines_matrix
    if hasattr(inst, "durations_matrix") and hasattr(inst, "machines_matrix"):
        dur = inst.durations_matrix
        mac = inst.machines_matrix
        return [
            [(mac[j][k], dur[j][k]) for k in range(len(dur[j]))]
            for j in range(len(dur))
        ]

    # 2) processing_times
    if hasattr(inst, "processing_times"):
        proc = inst.processing_times
        return [
            [(m, proc[j][m]) for m in range(len(proc[j]))] for j in range(len(proc))
        ]

    # 3) jobs
    if hasattr(inst, "jobs"):
        raw = inst.jobs
        # já em formato de tuplas
        if isinstance(raw, list) and raw and isinstance(raw[0][0], tuple):
            return raw
        # objetos com .machine e .time
        first = raw[0][0]
        if hasattr(first, "machine") and hasattr(first, "time"):
            return [[(t.machine, t.time) for t in job] for job in raw]
        # objetos com .machine_id e .duration
        if hasattr(first, "machine_id") and hasattr(first, "duration"):
            return [[(t.machine_id, t.duration) for t in job] for job in raw]

    raise AttributeError(
        "API desconhecida no JobShopInstance; não foi possível extrair jobs."
    )


def decode(jobs, seq):
    jnxt = [0] * len(jobs)
    jfin = [0] * len(jobs)
    num_machines = max(m for job in jobs for m, _ in job) + 1
    mfin = [0] * num_machines

    for j in seq:
        m, p = jobs[j][jnxt[j]]
        start = max(jfin[j], mfin[m])
        finish = start + p
        jfin[j] = finish
        mfin[m] = finish
        jnxt[j] += 1

    return max(jfin)

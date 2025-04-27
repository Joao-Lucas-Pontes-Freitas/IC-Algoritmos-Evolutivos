#!/usr/bin/env python3
# Title: A Hybrid Particle Swarm Optimization Algorithm Enhanced with Nonlinear Inertial Weight and Gaussian Mutation for Job Shop Scheduling Problems
# Link: https://www.mdpi.com/2227-7390/8/8/1355

import random
import argparse
import numpy as np

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
    if hasattr(inst, 'durations_matrix') and hasattr(inst, 'machines_matrix'):
        dur = inst.durations_matrix
        mac = inst.machines_matrix
        return [
            [(mac[j][k], dur[j][k]) for k in range(len(dur[j]))]
            for j in range(len(dur))
        ]

    # 2) processing_times
    if hasattr(inst, 'processing_times'):
        proc = inst.processing_times
        return [
            [(m, proc[j][m]) for m in range(len(proc[j]))]
            for j in range(len(proc))
        ]

    # 3) jobs
    if hasattr(inst, 'jobs'):
        raw = inst.jobs
        # já em formato de tuplas
        if isinstance(raw, list) and raw and isinstance(raw[0][0], tuple):
            return raw
        # objetos com .machine e .time
        first = raw[0][0]
        if hasattr(first, 'machine') and hasattr(first, 'time'):
            return [
                [(t.machine, t.time) for t in job]
                for job in raw
            ]
        # objetos com .machine_id e .duration
        if hasattr(first, 'machine_id') and hasattr(first, 'duration'):
            return [
                [(t.machine_id, t.duration) for t in job]
                for job in raw
            ]

    raise AttributeError("API desconhecida no JobShopInstance; não foi possível extrair jobs.")

def decode(jobs, seq):
    jnxt = [0] * len(jobs)
    jfin = [0] * len(jobs)
    num_machines = max(m for job in jobs for m,_ in job) + 1
    mfin = [0] * num_machines

    for j in seq:
        m, p = jobs[j][jnxt[j]]
        start = max(jfin[j], mfin[m])
        finish = start + p
        jfin[j] = finish
        mfin[m] = finish
        jnxt[j] += 1

    return max(jfin)

def pso(jobs, seed):
    random.seed(seed)
    np.random.seed(seed)

    numero_tarefas = sum(len(job) for job in jobs)
    ops = [j for j in range(len(jobs)) for _ in range(len(jobs[j]))]

    populacao = 100
    iters = 10000
    w_max, w_min = 0.9, 0.4
    c1 = c2 = 2.0

    particulas = np.random.rand(populacao, numero_tarefas)
    vel = np.random.uniform(-0.2, 0.2, (populacao, numero_tarefas))

    def eval_particle(p):
        seq = [op for _, op in sorted(zip(p, ops))]
        return decode(jobs, seq)

    pbest = particulas.copy()
    pbest_vals = np.array([eval_particle(p) for p in pbest])

    best_idx = pbest_vals.argmin()
    gbest = pbest[best_idx].copy()
    
    gbest_val = pbest_vals[best_idx]

    for t in range(iters):
        w = w_max - ((t * (w_max - w_min)) / iters) * np.sin((t*np.pi) / (2*iters))
        r1 = np.random.rand(populacao, numero_tarefas)
        r2 = np.random.rand(populacao, numero_tarefas)

        vel = w * vel + c1 * r1 * (pbest - particulas) + c2 * r2 * (gbest - particulas)
        particulas = particulas + vel

        noise = np.random.normal(0, 1, particulas.shape)
        mutaded = particulas + noise

        fitness_atual = np.array([eval_particle(p) for p in particulas])
        fitness_novo = np.array([eval_particle(p) for p in mutaded])

        particulas[fitness_novo < fitness_atual] = mutaded[fitness_novo < fitness_atual]

        vals = np.array([eval_particle(p) for p in particulas])
        better = vals < pbest_vals

        pbest[better] = particulas[better]
        pbest_vals[better] = vals[better]

        idx_min = pbest_vals.argmin()

        if pbest_vals[idx_min] < gbest_val:
            gbest = pbest[idx_min].copy()
            gbest_val = pbest_vals[idx_min]

    return gbest_val

def main():
    parser = argparse.ArgumentParser(description="Compare GA vs PSO on JSSP benchmarks")
    parser.add_argument("--instance", default="ft06", help="nome do benchmark (e.g., abz7)")
    parser.add_argument("--seed", type=int, default=42, help="semente para aleatoriedade")
    args = parser.parse_args()

    jobs = load_instance(args.instance.lower())

    experimentos = 1
    resultados_pso = np.zeros(experimentos)

    for i in range(experimentos):
        pso_val = pso(jobs, args.seed + i)
        resultados_pso[i] = pso_val
 
    print("PSO: ", np.mean(resultados_pso))
    print("PSO Melhor: ", np.min(resultados_pso))
    print("PSO Pior: ", np.max(resultados_pso))

if __name__ == "__main__":
    main()
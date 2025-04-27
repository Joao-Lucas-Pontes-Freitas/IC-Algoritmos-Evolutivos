#!/usr/bin/env python3
# Title: A Modified Genetic Algorithm with Local Search Strategies and Multi-Crossover Operator for Job Shop Scheduling Problem
# Link: https://www.mdpi.com/1424-8220/20/18/5440
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

def random_seq(jobs):
    seq = [j for j in range(len(jobs)) for _ in range(len(jobs[j]))]
    random.shuffle(seq)
    return seq

def ox(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]

    total = {}
    for x in p1: total[x] = total.get(x, 0) + 1
    seg = {}
    for x in p1[a:b]: seg[x] = seg.get(x, 0) + 1
    missing = {j: total[j] - seg.get(j, 0) for j in total}

    fill = []
    for x in p2:
        if missing.get(x, 0) > 0:
            fill.append(x)
            missing[x] -= 1

    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1

    return child

def pmx(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]

    total = {}
    for x in p1: total[x] = total.get(x, 0) + 1
    seg = {}
    for x in p1[a:b]: seg[x] = seg.get(x, 0) + 1
    missing = {j: total[j] - seg.get(j, 0) for j in total}

    fill = []
    for x in p2:
        if missing.get(x, 0) > 0:
            fill.append(x)
            missing[x] -= 1

    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1

    return child

def pox(p1, p2):
    n = len(p1)
    mask = [random.random() < 0.5 for _ in range(n)]
    child = [p1[i] if mask[i] else None for i in range(n)]

    total = {}
    for x in p1: total[x] = total.get(x, 0) + 1
    seg = {}
    for i, use in enumerate(mask):
        if use:
            seg[p1[i]] = seg.get(p1[i], 0) + 1
    missing = {j: total[j] - seg.get(j, 0) for j in total}

    fill = []
    for x in p2:
        if missing.get(x, 0) > 0:
            fill.append(x)
            missing[x] -= 1

    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1

    return child

def ga(jobs, seed):
    random.seed(seed)
    np.random.seed(seed)

    pop_size = 100
    gens = 100
    mut_rate = 0.3

    pop = [random_seq(jobs) for _ in range(pop_size)]
    best = min(pop, key=lambda s: decode(jobs, s))
    best_val = decode(jobs, best)

    for g in range(gens):
        new_pop = []
        for _ in range(pop_size // 2):
            def select_one():
                a, b = random.sample(pop, 2)
                return a if decode(jobs, a) < decode(jobs, b) else b

            p1 = select_one()
            p2 = select_one()
            cx = random.choice([ox, pmx, pox])
            for parent in ((p1, p2), (p2, p1)):
                child = cx(*parent)
                if random.random() < mut_rate:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                new_pop.append(child)

        pop = new_pop
        if g % 50 == 49:
            for ind in pop:
                for _ in range(20):
                    i, j = sorted(random.sample(range(len(ind)), 2))
                    ind[i:j] = list(reversed(ind[i:j]))

        current = min(pop, key=lambda s: decode(jobs, s))
        current_val = decode(jobs, current)
        if current_val < best_val:
            best, best_val = current, current_val

    return int(best_val)

def pso(jobs, seed):
    random.seed(seed)
    np.random.seed(seed)

    n_ops = sum(len(job) for job in jobs)
    ops = [j for j in range(len(jobs)) for _ in range(len(jobs[j]))]

    part = 50
    iters = 1000
    w_max, w_min = 0.9, 0.4
    c1 = c2 = 2.0
    mut_prob = 0.05

    pos = np.random.rand(part, n_ops)
    vel = np.random.rand(part, n_ops) - 0.5

    def eval_particle(p):
        seq = [op for _, op in sorted(zip(p, ops))]
        return decode(jobs, seq)

    pbest = pos.copy()
    pbest_vals = np.array([eval_particle(p) for p in pbest])
    best_idx = pbest_vals.argmin()
    gbest = pbest[best_idx].copy()
    gbest_val = pbest_vals[best_idx]

    for t in range(iters):
        w = w_max - (w_max - w_min) * (t / iters) ** 2
        r1 = np.random.rand(part, n_ops)
        r2 = np.random.rand(part, n_ops)

        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = pos + vel

        mask = np.random.rand(part) < mut_prob
        noise = np.random.normal(0, 1, size=pos[mask].shape)
        pos[mask] += noise

        vals = np.array([eval_particle(p) for p in pos])
        better = vals < pbest_vals
        pbest[better] = pos[better]
        pbest_vals[better] = vals[better]

        idx_min = pbest_vals.argmin()
        if pbest_vals[idx_min] < gbest_val:
            gbest = pbest[idx_min].copy()
            gbest_val = pbest_vals[idx_min]

    return int(gbest_val)

def main():
    parser = argparse.ArgumentParser(description="Compare GA vs PSO on JSSP benchmarks")
    parser.add_argument("--instance", default="ft06", help="nome do benchmark (e.g., abz7)")
    parser.add_argument("--seed", type=int, default=42, help="semente para aleatoriedade")
    args = parser.parse_args()

    jobs = load_instance(args.instance.lower())

    experimentos = 30
    resultados_ga = np.zeros(experimentos)
    resultados_pso = np.zeros(experimentos)

    for i in range(experimentos):
        ga_val = ga(jobs, args.seed + i)
        pso_val = pso(jobs, args.seed + i)
        resultados_ga[i] = ga_val
        resultados_pso[i] = pso_val
 
    print("GA: ", np.mean(resultados_ga))
    print("PSO: ", np.mean(resultados_pso))

if __name__ == "__main__":
    main()
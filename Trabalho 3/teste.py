# Title: A Modified Genetic Algorithm with Local Search Strategies and Multi-Crossover Operator for Job Shop Scheduling Problem
# Link: https://www.mdpi.com/1424-8220/20/18/5440
# Title: A Hybrid Particle Swarm Optimization Algorithm Enhanced with Nonlinear Inertial Weight and Gaussian Mutation for Job Shop Scheduling Problems
# Link: https://www.mdpi.com/2227-7390/8/8/1355


# utils_benchmark.py
from job_shop_lib.benchmarking import (
    load_benchmark_instance,
)  # :contentReference[oaicite:0]{index=0}
import random, numpy as np




def load_instance(name: str):
    """
    Carrega um benchmark JSSP do job-shop-lib
    e devolve no formato [[(machine, duration), …], …].
    """
    inst = load_benchmark_instance(name)  # JobShopInstance
    jobs: list[list[tuple[int, int]]] = []

    for job in inst.jobs:  # cada job é uma lista de Operation
        ops = []
        for op in job:
            machine = op.machine_id  # uma única máquina por operação
            duration = op.duration
            ops.append((machine, duration))
        jobs.append(ops)

    return jobs


# ===  Escolha o benchmark  ===============================
instance_name = "abz7"  # mude para "la21", "abz7", "orb5", "yn4"...
jobs = load_instance(instance_name)
# =========================================================

machines = sorted({m for job in jobs for m, _ in job})
job_count = {j: len(jobs[j]) for j in range(len(jobs))}
total_ops = sum(job_count.values())
# … (restante do seu código GA/PSO permanece igual) …

ops_list = []
for j in range(len(jobs)):
    for k in range(job_count[j]):
        ops_list.append(j)


def decode_sequence(jobs, sequence):
    job_next_op = [0] * len(jobs)
    job_finish = [0] * len(jobs)
    machine_finish = {m: 0 for m in machines}
    for job_id in sequence:
        op_index = job_next_op[job_id]
        machine, proc = jobs[job_id][op_index]
        start = max(job_finish[job_id], machine_finish[machine])
        finish = start + proc
        job_finish[job_id] = finish
        machine_finish[machine] = finish
        job_next_op[job_id] += 1
    return max(job_finish)


def ga_optimize(jobs, pop_size=50, generations=100):
    def random_sequence():
        seq = []
        for job_id, count in job_count.items():
            seq += [job_id] * count
        random.shuffle(seq)
        return seq

    def crossover(seq1, seq2):
        n = len(seq1)
        cp = random.randint(1, n - 1)
        child = [-1] * n
        child[:cp] = seq1[:cp]
        needed = {j: job_count[j] for j in job_count}
        for j in child[:cp]:
            needed[j] -= 1
        idx = cp
        for x in seq2:
            if idx >= n:
                break
            if needed[x] > 0:
                child[idx] = x
                needed[x] -= 1
                idx += 1
        for i in range(n):
            if child[i] == -1:
                for j in needed:
                    if needed[j] > 0:
                        child[i] = j
                        needed[j] -= 1
                        break
        return child

    def mutate(seq, rate=0.1):
        if random.random() < rate:
            i, j = random.sample(range(len(seq)), 2)
            seq[i], seq[j] = seq[j], seq[i]
        return seq

    pop = [random_sequence() for _ in range(pop_size)]
    best = min(pop, key=lambda s: decode_sequence(jobs, s))
    best_val = decode_sequence(jobs, best)
    for _ in range(generations):
        fitness = [decode_sequence(jobs, ind) for ind in pop]
        new_pop = []
        for _ in range(pop_size // 2):

            def tournament():
                i, j = random.sample(range(pop_size), 2)
                return pop[i] if fitness[i] < fitness[j] else pop[j]

            p1 = tournament()
            p2 = tournament()
            c1 = crossover(p1, p2)
            c2 = crossover(p2, p1)
            new_pop.append(mutate(c1))
            new_pop.append(mutate(c2))
        pop = new_pop
        for ind in pop:
            val = decode_sequence(jobs, ind)
            if val < best_val:
                best_val = val
                best = ind.copy()
    return best, best_val


def pso_optimize(jobs, num_particles=30, iterations=100):
    global ops_list
    ops_list = []
    for j in range(len(jobs)):
        for k in range(job_count[j]):
            ops_list.append(j)
    n = total_ops
    pbest_pos = []
    pbest_val = []
    particles_pos = []
    particles_vel = []
    for _ in range(num_particles):
        pos = np.random.rand(n).tolist()
        vel = (np.random.rand(n) - 0.5).tolist()
        particles_pos.append(pos)
        particles_vel.append(vel)
        seq = [job for _, job in sorted(zip(pos, ops_list))]
        val = decode_sequence(jobs, seq)
        pbest_pos.append(pos.copy())
        pbest_val.append(val)
    gbest_index = min(range(num_particles), key=lambda i: pbest_val[i])
    gbest_pos = pbest_pos[gbest_index].copy()
    gbest_val = pbest_val[gbest_index]
    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.5, 1.5
    for t in range(iterations):
        w = w_max - (w_max - w_min) * (t / iterations) ** 2
        for i in range(num_particles):
            pos = particles_pos[i]
            vel = particles_vel[i]
            for d in range(n):
                r1 = random.random()
                r2 = random.random()
                vel[d] = (
                    w * vel[d]
                    + c1 * r1 * (pbest_pos[i][d] - pos[d])
                    + c2 * r2 * (gbest_pos[d] - pos[d])
                )
                pos[d] += vel[d]
            if random.random() < 0.1:
                for d in range(n):
                    pos[d] += np.random.normal(0, 1)
            seq = [job for _, job in sorted(zip(pos, ops_list))]
            val = decode_sequence(jobs, seq)
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = pos.copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = pos.copy()
    best_seq = [job for _, job in sorted(zip(gbest_pos, ops_list))]
    return best_seq, gbest_val


if __name__ == "__main__":
    medias_ga = []
    medias_pso = []
    experimentos = 30
    for i in range(experimentos):
        best_seq_ga, best_val_ga = ga_optimize(jobs)
        medias_ga.append(best_val_ga)

        best_seq_pso, best_val_pso = pso_optimize(jobs)
        medias_pso.append(best_val_pso)
    print("GA:", sum(medias_ga) / len(medias_ga))
    print("PSO:", sum(medias_pso) / len(medias_pso))

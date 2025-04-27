import numpy as np
from benchmark import decode


def pso(jobs, seed):
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
        w = w_max - ((t * (w_max - w_min)) / iters) * np.sin((t * np.pi) / (2 * iters))
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

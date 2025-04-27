# Title: A Hybrid Particle Swarm Optimization Algorithm Enhanced with Nonlinear Inertial Weight and Gaussian Mutation for Job Shop Scheduling Problems
# Link: https://www.mdpi.com/2227-7390/8/8/1355

import argparse
import numpy as np
from benchmark import load_instance
from pso import pso


def main():
    benchmarks = ["abz5", "abz6", "abz7", "abz8", "abz9", "ft06", "ft10", "ft20", "yn1", "yn2", "yn3", "yn4"]
    print('-' * 30)
    print()

    for benchmark in benchmarks:
        
        print(f"Executando benchmark: {benchmark}")
        
        jobs = load_instance(benchmark)

        seed = 42
        experimentos = 1
        resultados_pso = np.zeros(experimentos)

        for i in range(experimentos):
            pso_val = pso(jobs, seed + i)
            resultados_pso[i] = pso_val

        print("PSO - Média: ", np.mean(resultados_pso))
        print("PSO - Melhor: ", np.min(resultados_pso))
        print("PSO - Pior: ", np.max(resultados_pso))
        print()
        print("GA - Média: ", np.mean(resultados_pso))
        print("GA - Melhor: ", np.min(resultados_pso))
        print("GA - Pior: ", np.max(resultados_pso))
        print('-' * 30)
        print()


if __name__ == "__main__":
    main()

# Title: A Hybrid Particle Swarm Optimization Algorithm Enhanced with Nonlinear Inertial Weight and Gaussian Mutation for Job Shop Scheduling Problems
# Link: https://www.mdpi.com/2227-7390/8/8/1355

import numpy as np
import sys
from benchmark import load_instance
from pso import pso
from ga import ga


def main():
    with open('resultados.txt', 'w') as sys.stdout:
        benchmarks = ["abz5", "abz6", "abz7", "abz8", "abz9", "ft06", "ft10", "ft20", "yn1", "yn2", "yn3", "yn4"]
        print('-' * 30)
        print()

        for benchmark in benchmarks:
            
            print(f"Executando benchmark: {benchmark}")
            
            jobs = load_instance(benchmark)

            seed = 42
            experimentos = 30
            resultados_pso = np.zeros(experimentos)
            resultados_ga = np.zeros(experimentos)

            for i in range(experimentos):
                pso_val = pso(jobs, seed + i)
                ga_val = ga(jobs, seed + i)
                resultados_pso[i] = pso_val
                resultados_ga[i] = ga_val

            print("PSO - Media: ", np.mean(resultados_pso))
            print("PSO - Melhor: ", np.min(resultados_pso))
            print("PSO - Pior: ", np.max(resultados_pso))
            print()
            print("GA - Media: ", np.mean(resultados_ga))
            print("GA - Melhor: ", np.min(resultados_ga))
            print("GA - Pior: ", np.max(resultados_ga))
            print('-' * 30)
            print()

if __name__ == "__main__":
    main()

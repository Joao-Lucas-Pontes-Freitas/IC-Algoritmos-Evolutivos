# Title: A Hybrid Particle Swarm Optimization Algorithm Enhanced with Nonlinear Inertial Weight and Gaussian Mutation for Job Shop Scheduling Problems
# Link: https://www.mdpi.com/2227-7390/8/8/1355

import argparse
import numpy as np
from benchmark import load_instance
from pso import pso


def main():
    parser = argparse.ArgumentParser(description="Compare GA vs PSO on JSSP benchmarks")
    parser.add_argument(
        "--instance", default="ft06", help="nome do benchmark (e.g., abz7)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="semente para aleatoriedade"
    )
    args = parser.parse_args()

    jobs = load_instance(args.instance.lower())

    experimentos = 30
    resultados_pso = np.zeros(experimentos)

    for i in range(experimentos):
        pso_val = pso(jobs, args.seed + i)
        resultados_pso[i] = pso_val

    print("PSO: ", np.mean(resultados_pso))
    print("PSO Melhor: ", np.min(resultados_pso))
    print("PSO Pior: ", np.max(resultados_pso))


if __name__ == "__main__":
    main()

import numpy as np
from algorithms import calcula_fitness


class Particle:
    def __init__(self, position):
        self.position = position
        self.pbest = position.copy()
        self.velocity = np.zeros(len(position))


def evolve(particles, workflow, vms):
    def calc_fit(pos):
        return calcula_fitness(pos, workflow, vms)

    gbest = particles[0].position.copy()

    for i in range(len(particles)):
        pbest_fit = calc_fit(particles[i].pbest)
        curr_fit = calc_fit(particles[i].position)
        gbest_fit = calc_fit(gbest)

        if pbest_fit > curr_fit:
            particles[i].pbest = particles[i].position.copy()
            pbest_fit = curr_fit

        if gbest_fit > pbest_fit:
            gbest = particles[i].pbest.copy()

    return gbest


def update_velocity(particles, gbest, C1=1.0, C2=1.1):
    r1 = np.random.uniform(0, 1 + np.finfo(float).eps)
    r2 = np.random.uniform(0, 1 + np.finfo(float).eps)

    for k in range(len(particles)):
        if particles[k].position == particles[k].pbest:
            particles[k].velocity -= C1 * r1
        else:
            particles[k].velocity += C1 * r1

        if particles[k].position == gbest:
            particles[k].velocity -= C2 * r2
        else:
            particles[k].velocity += C2 * r2


def update_position(particles):
    def swap(a, b):
        a, b = b, a

    for j in range(len(particles)):
        max1 = np.argmax(particles[j].velocity)
        og_value = particles[j].velocity[max1]

        particles[j].velocity[max1] = -np.inf

        max2 = np.argmax(particles[j].velocity)
        particles[j].velocity[max1] = og_value

        swap(particles[j].position[max1], particles[j].position[max2])


def pso(particles, workflow, vms, num_iterations):
    gbest = particles[0].position.copy()

    for _ in range(num_iterations):
        gbest = evolve(particles, workflow, vms)
        update_velocity(particles, gbest)
        update_position(particles)

    return gbest, calcula_fitness(gbest, workflow, vms)


# pop_particulas = [Particula(gerar_individuo(num_tarefas, num_vms)) for _ in range(pop_size)]

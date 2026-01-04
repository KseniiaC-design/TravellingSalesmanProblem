import pandas as pd
import numpy as np
from numpy.random import randint, rand, permutation, seed
from plot import plot_route_ga


df = pd.read_csv("Gemeinden.csv", sep=";", encoding="cp1252", decimal=",")

print(list(df.columns))

df = df.head(30)

city_names = df["city"].tolist()
latitudes  = df["lat"].tolist()
longitudes = df["lon"].tolist()

# zip pairs each entry in latitudes with the respective entry in longitudes
cities = list(zip(latitudes, longitudes))

start_city = city_names[0]

# The haversine formula determines the distance between two points on a sphere given their longitudes and latitudes.
def haversine(coord1, coord2):
    R = 6371 #Radius
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def total_distance(route, cities):
    distance = 0.0
    for i in range(len(route)):
        a = cities[route[i]]
        b = cities[route[(i + 1) % len(route)]]
        distance += haversine(a, b)
    return distance

def initial_population(n_pop, n_cities):
    return [permutation(n_cities).tolist() for _ in range(n_pop)]

# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    i = randint(len(pop))
    # check if better (e.g. perform a tournament)
    for j in randint(0, len(pop), k-1):
        if scores[j] < scores[i]:
            i = j
    return pop[i]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    
    # children are copies of parents
    if rand() >= r_cross:
        return [p1.copy(), p2.copy()]

    #splits at random points (will be the same as parents)
    size = len(p1)
    a, b = sorted(randint(0, size, 2))

    def ox(parent1, parent2):
        #empty route
        child = [None]*size
        #copies cities from parent1
        child[a:b] = parent1[a:b]
        ptr = b
        #adds cities from parent2
        for c in parent2:
            if c not in child:
                if ptr >= size:
                    ptr = 0
                child[ptr] = c
                ptr += 1
        return child

    return [ox(p1, p2), ox(p2, p1)]

# swap two random cities
def mutation(route, r_mut):
    for i in range(len(route)):
        if rand() < r_mut:
            j = randint(len(route))
            route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm_tsp(cities, n_iter, n_pop, r_cross, r_mut):
    pop = initial_population(n_pop, len(cities))

    best, best_eval = pop[0], total_distance(pop[0], cities)

    for gen in range(n_iter):
        #calculate fitness 
        scores = [total_distance(p, cities) for p in pop]

        #update best 
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(f"Gen {gen}: neue beste Distanz = {best_eval:.2f} km")

        #tournament selection
        selected = [selection(pop, scores) for _ in range(n_pop)]

        #crossover and mutation
        children = []
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            for c in crossover(p1, p2, r_cross):
                children.append(mutation(c, r_mut))

        pop = children

    return best, best_eval

seed(1)
#cities=cities
n_iter=500
n_pop=100
r_cross=0.9
r_mut=0.02

best_route, best_distance = genetic_algorithm_tsp(cities, n_iter, n_pop, r_cross, r_mut)

print("\nOptimale Route:")
print(start_city)
for i in best_route:
    print(city_names[i])
print(start_city)
print(f"\nGesamtdistanz: {best_distance:.2f} km")

plot_route_ga(
    best_route,
    city_names,
    latitudes,
    longitudes,
    title=f"Optimale TSP-Route (Distanz: {best_distance:.2f} km)"
)




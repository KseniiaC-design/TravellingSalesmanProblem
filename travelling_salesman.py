import random
import pandas as pd
import numpy as np
from numpy.random import randint, rand, permutation, seed
from plot import plot_route_ga
from plot_zoom import plot_route_ga_zoom
import time
from itertools import permutations


#Loading and preprocessing city data from CSV file
#This CSV file was cleaned and prepared by AI
def load_and_validate_csv(path):
    #Error handling for file not found
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError("City data file not found.")

    # Select only relevant columns and rename them
    df = df.iloc[:, [7, 13, 14]]
    df.columns = ["city", "lon", "lat"]


    # Clean selected columns
    # Devide the city name (as string) into before and after comma and keep first part (Berlin, Germany -> Berlin)
    #Convert longitude and latitude to numeric (float), if number not convertable/coerce to NaN
    df["city"] = df["city"].astype(str).str.split(",").str[0].str.strip()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    # Remove rows with missing or duplicate data
    df = df.dropna()
    df = df.drop_duplicates(subset="city") #Remove rows with duplicate city names

    # Reset index after cleaning (because rows were removed)
    return df.reset_index(drop=True)





# The haversine formula determines the distance between two points on a sphere given their longitudes and latitudes.
def haversine(coord1, coord2):
    R = 6371 #Radius
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def total_distance(route, cities, start_city=0):
     # Calculates total tour length: start → cities in route → start
    full_route = [start_city] + route + [start_city]
    return sum(
        #assign coordinates of cities in the route to haversine function to calculate distance
        #iterate through all cities in the full route, excluding the last one
        #Meaning: We calculate distance between each pair of consecutive cities in the full route
        haversine(cities[full_route[i]], cities[full_route[i + 1]])
        for i in range(len(full_route) - 1) 
    )

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

#Exact TSP solution for small number of cities (≤ 10) using brute-force for comparison
def exact_tsp(coords, start_city):

    #Generate all possible routes through cities excluding start city
    cities = [i for i in range(len(coords)) if i != start_city]

    #Keep track of best route and its distance
    best_route = None
    best_distance = float("inf")

    #Iterate through all permutations of cities
    for perm in permutations(cities):
        #Calculate distance of the current route
        dist = total_distance(list(perm), coords, start_city)
        #Update best route if current one is better
        if dist < best_distance:
            best_distance = dist
            best_route = list(perm)
    #Return best route and its distance
    return best_route, best_distance

#Main execution
if __name__ == "__main__":
    random.seed(1) #For reproducibility

    # Load and preprocess city data
    df = load_and_validate_csv('AuszugGV3QAktuell_clean.csv')
    df = df.head(30)

    city_names = df["city"].tolist()
    latitudes = df["lat"].tolist()
    longitudes = df["lon"].tolist()
    cities = list(zip(latitudes, longitudes))
    start_city = city_names[0]



#Time and run Genetic Algorithm
    t0 = time.time()
    #Run genetic algorithm and get best route and its distance
    best_route, best_distance = genetic_algorithm_tsp(cities, n_iter=500, n_pop=100, r_cross=0.9, r_mut=0.02)
    ga_time = time.time() - t0 #Calculate runtime

    #Print results for GA 
    print("\n--- Optimal Route (GA) ---")
    print("\nRoute:")
    for i in [0] + best_route + [0]:
        print(city_names[i])


    print("\n--- Metrics for Optimal GA solution ---")
    print(f"GA distance: {best_distance:.2f} km") #Genetic Algorithm overall distance
    print(f"GA runtime: {ga_time:.3f} s") #Genetic Algorithm runtime


    #Now run GA and exact TSP for comparison
    #Only use first 8 cities for exact solution due to high computational cost
    df_small = df.head(8)
    city_names_small = df_small["city"].tolist()
    latitudes_small = df_small["lat"].tolist()
    longitudes_small = df_small["lon"].tolist()
    coords_small = list(zip(latitudes_small, longitudes_small))
    start_city_small = 0  # Use index

    #Time and run exact TSP 
    t0 = time.time()
    exact_route, exact_distance = exact_tsp(coords_small, start_city_small)
    exact_time = time.time() - t0


    #Print Comparision results
    print("\n--- Optimal Route für Subset (Exact) ---")
    print("\nRoute:")
    for i in [0] + exact_route + [0]:
        print(city_names_small[i])


#Time and run Genetic Algorithm on small dataset
    print("\n--- Optimal Route für Subset (GA) ---")
    t0 = time.time()
    ga_route_small, ga_distance_small = genetic_algorithm_tsp(coords_small, n_iter=500, n_pop=100, r_cross=0.9, r_mut=0.02)
    ga_time_small = time.time() - t0
    print("\nRoute:")
    for i in [0] + ga_route_small + [0]:
        print(city_names_small[i])
    
    print("\n--- Metrics Comparison for Subset ---")
    print(f"GA distance: {ga_distance_small:.2f} km")
    print(f"Exact distance: {exact_distance:.2f} km")
    print(f"GA runtime: {ga_time_small:.3f} s")
    print(f"Exact runtime: {exact_time:.4f} s")




#Figure plotting
plot_route_ga(
    best_route,
    city_names,
    latitudes,
    longitudes,
    title=f"Optimale TSP-Route (Distanz: {best_distance:.2f} km)"
)

plot_route_ga_zoom(
    best_route,
    city_names,
    latitudes,
    longitudes,
    title="Zoom: Route im unteren linken Cluster",
    lon_min = 8.85,
    lon_max = 9.35,
    lat_min = 53.88,
    lat_max = 54.12
)





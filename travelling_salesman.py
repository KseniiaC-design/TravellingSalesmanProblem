import pandas as pd
import numpy as np
from numpy.random import randint, rand, permutation
from plot import plot_route_ga
from plot_zoom import plot_route_ga_zoom
import time
from itertools import permutations


#Loading and preprocessing city data from CSV file
#This CSV file was cleaned and prepared by AI
def load_and_validate_csv(path):
    #Error handling for file not found
    #If found load into DataFrame
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError("City data file not found.")

    # Select only relevant columns and rename them
    df = df.iloc[:, [7, 13, 14]]
    df.columns = ["city", "lon", "lat"]


    # Clean selected columns
    # Devide the city name (as string) into before and after comma and keep first part (Berlin, Germany -> Berlin)
    #Convert longitude and latitude to numeric (float), if number not convertable/theres problems coerce to NaN
    df["city"] = df["city"].astype(str).str.split(",").str[0].str.strip()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    # Remove rows with missing or duplicate data
    df = df.dropna()
    df = df.drop_duplicates(subset="city") #Remove rows with duplicate city names

    # Reset index after cleaning (because rows were removed)
    return df.reset_index(drop=True)





# The haversine formula determines the distance between two points on a sphere given their longitudes and latitudes.
# coord1 contains the latitude and longitude of the first city,
# coord2 contains the latitude and longitude of the second city
def haversine(coord1, coord2):
    R = 6371 #Radius of globe in kilometers

    #Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    #Difference between cities
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    #Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    #return distance in kilometers
    return 2 * R * np.arcsin(np.sqrt(a))

# Calculates total tour length: 
def total_distance(route, cities, start_city=0):
     #defines route as start city → cities in route → start city
    full_route = [start_city] + route + [start_city]
    
    #sum distances between consecutive cities in the full route
    return sum(
        #assign coordinates of cities in the route to haversine function to calculate distance
        #iterate through all cities in the full route, excluding the last one
        #Meaning: We calculate distance between each pair of consecutive cities in the full route
        haversine(cities[full_route[i]], cities[full_route[i + 1]])
        for i in range(len(full_route) - 1) 
    )

# generate initial population
def initial_population(n_pop, n_cities, start_city=0):
    # generate random permutations meaning order of cities excluding the start city
    cities = [i for i in range(n_cities) if i != start_city]
    return [permutation(cities).tolist() for _ in range(n_pop)]

# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    i = randint(len(pop))
    # check if better , meaning lower distance
    for j in randint(0, len(pop), k-1):
        if scores[j] < scores[i]:
            i = j
    #return best individual
    return pop[i] 

# crossover two parents to create two children
def crossover(p1, p2, r_cross=0.9): #Crossover-Probability (90 %)
    
    if rand() >= r_cross: #No crossover for 10% of the time, return copy of parents 
        return [p1.copy(), p2.copy()]


    #size of the route of the parent1
    size = len(p1)
    a, b = sorted(randint(0, size, 2)) #random crossover points

# order crossover (OX) implementation, since we are dealing with permutations (routes)
        #Create child by copying a slice from parent1 and filling remaining positions with parent2
    def ox(parent1, parent2):
        #empty route for child initialized with None and size of parent
        child = [None]*size
        #copies cities from parent1
        child[a:b] = parent1[a:b]
        
        
        #Save genes/cities already used in the child
        used_cities = set(child[a:b])
        #Extract  genes from parent2 in order, excluding already used genes
        fill = [gene for gene in p2 if gene not in used_cities]

        #Fill remaining positions in child with genes from fill list
        idx = 0
        for i in range(len(child)):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1

        return child
    #Create two children using OX, with parents swapped
    return [ox(p1, p2), ox(p2, p1)]

# von hier
# swap two random cities
def mutation(route, r_mut):
    for i in range(len(route)): #iterates over every city
        if rand() < r_mut:#r_mut=0.02
            j = randint(len(route)) #randomly chooses 2. position
            route[i], route[j] = route[j], route[i]#replaces the two positions
    return route

# genetic algorithm for TSP, Main logic
def genetic_algorithm_tsp(cities, n_iter, n_pop, r_cross, r_mut):
    #create initial population with start city as index 0
    pop = initial_population(n_pop, len(cities), start_city=0)

    #initialize best solution
    best, best_eval = pop[0], total_distance(pop[0], cities)

    #evolve population over fixed number of generations
    for gen in range(n_iter):
        #calculate fitness scores for each individual in population
        scores = [total_distance(p, cities) for p in pop]

        #update best 
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                #print progress, always when a new best solution is found
                print(f"Gen {gen}: neue beste Distanz = {best_eval:.2f} km")

        #Selects n_pop parents using tournament selection, favoring shorter routes while keeping diversity
        selected = [selection(pop, scores) for _ in range(n_pop)]

        #crossover and mutation
        children = [] #new empty population for children
        for i in range(0, n_pop, 2): #iterate through selected parents in pairs
            p1, p2 = selected[i], selected[i+1]#choose two parents for crossover
            for c in crossover(p1, p2, r_cross): #crossover parents to create children
                children.append(mutation(c, r_mut)) #mutate children and add to new population  

        pop = children #replace old population with new one 

    return best, best_eval #return best route and its distance

#Exact TSP solution for small number of cities using brute-force for comparison
def exact_tsp(coords, start_city):#evaluates all permutations

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
    np.random.seed(1) #For reproducibility

    # Load and preprocess city data
    df = load_and_validate_csv('AuszugGV3QAktuell_clean.csv')
    df = df.head(30) #Use only first 30 cities for GA
   
    latitudes = df["lat"].tolist() #List of latitudes
    longitudes = df["lon"].tolist() #List of longitudes
    cities = list(zip(latitudes, longitudes)) #List of tuples of cities with (latitude, longitude)
    city_names = df["city"].tolist() #List of city names used for printing results
    

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
#bis hier

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
    title="Zoom: Route im unteren rechten Cluster",
    lon_min = 25600,
    lon_max = 25750,
    lat_min = 9.40,
    lat_max = 8.75
)





import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import lru_cache

# TSP Problem Configuration
def load_dataset(val):
    if val == 1:
        Prob = 'Eil51'
        x_coords = [37, 49, 52, 20, 40, 21, 17, 31, 52, 51, 42, 31, 5, 12, 36, 52, 27, 17, 13, 57, 62, 42, 16, 8, 7, 27,
                    30, 43,
                    58, 58, 37, 38, 46, 61, 62, 63, 32, 45, 59, 5, 10, 21, 5, 30, 39, 32, 25, 25, 48, 56, 30]
        y_coords = [52, 49, 64, 26, 30, 47, 63, 62, 33, 21, 41, 32, 25, 42, 16, 41, 23, 33, 13, 58, 42, 57, 57, 52, 38,
                    68, 48,
                    67, 48, 27, 69, 46, 10, 33, 63, 69, 22, 35, 15, 6, 17, 10, 64, 15, 10, 39, 32, 55, 28, 37, 40]
        #optimal_cost = 426
    elif val == 2:
        Prob = 'Berlin52'
        x_coords = [565, 25, 345, 945, 845, 880, 25, 525, 580, 650, 1605, 1220, 1465, 1530, 845, 725, 145, 415, 510,
                    560, 300, 520, 480, 835, 975, 1215, 1320, 1250, 660, 410, 420, 575, 1150, 700, 685, 685, 770, 795,
                    720, 760, 475, 95, 875, 700, 555, 830, 1170, 830, 605, 595, 1340, 1740]
        y_coords = [575, 185, 750, 685, 655, 660, 230, 1000, 1175, 1130, 620, 580, 200, 5, 680, 370, 665, 635, 875, 365,
                    465, 585, 415, 625, 580, 245, 315, 400, 180, 250, 555, 665, 1160, 580, 595, 610, 610, 645, 635, 650,
                    960, 260, 920, 500, 815, 485, 65, 610, 625, 360, 725, 245]
        #optimal_cost = 7542
    elif val == 3:
        Prob = 'St70'
        x_coords = [64, 80, 69, 72, 48, 58, 81, 79, 30, 42, 7, 29, 78, 64, 95, 57, 40, 68, 92, 62, 28, 76, 67, 93, 6,
                    87, 30,
                    77, 78, 55, 82, 73, 20, 27, 95, 67, 48, 75, 8, 20, 54, 63, 44, 52, 12, 25, 58, 5, 90, 41, 25, 37,
                    56, 10,
                    98, 16, 89, 48, 81, 29, 17, 5, 79, 9, 17, 74, 10, 48, 83, 84]
        y_coords = [96, 39, 23, 42, 67, 43, 34, 17, 23, 67, 76, 51, 92, 8, 57, 91, 35, 40, 34, 1, 43, 73, 88, 54, 8, 18,
                    9, 13,
                    94, 3, 88, 28, 55, 43, 86, 99, 83, 81, 19, 18, 38, 36, 33, 18, 13, 5, 85, 67, 9, 76, 76, 64, 63, 55,
                    7, 74,
                    60, 82, 76, 60, 22, 45, 70, 100, 82, 67, 68, 19, 86, 94]
        #optimal_cost = 675
    elif val == 4:
        Prob = 'Eil76'
        x_coords = [22, 36, 21, 45, 55, 33, 50, 55, 26, 40, 55, 35, 62, 62, 62, 21, 33, 9, 62, 66, 44, 26, 11, 7, 17,
                    41, 55,
                    35, 52, 43, 31, 22, 26, 50, 55, 54, 60, 47, 30, 30, 12, 15, 16, 21, 50, 51, 50, 48, 12, 15, 29, 54,
                    55, 67,
                    10, 6, 65, 40, 70, 64, 36, 30, 20, 15, 50, 57, 45, 38, 50, 66, 59, 35, 27, 40, 40, 40]
        y_coords = [22, 26, 45, 35, 20, 34, 50, 45, 59, 66, 65, 51, 35, 57, 24, 36, 44, 56, 48, 14, 13, 13, 28, 43, 64,
                    46, 34,
                    16, 26, 26, 76, 53, 29, 40, 50, 10, 15, 66, 60, 50, 17, 14, 19, 48, 30, 42, 15, 21, 38, 56, 39, 38,
                    57, 41,
                    70, 25, 27, 60, 64, 4, 6, 20, 30, 5, 70, 72, 42, 33, 4, 8, 5, 60, 24, 20, 37, 40]
        #optimal_cost = 538
    elif val == 5:
        Prob = 'Pr76'
        x_coords = [3600, 3100, 4700, 5400, 5608, 4493, 3600, 3100, 4700, 5400, 5610, 4492, 3600, 3100, 4700, 5400,
                    6650, 7300,
                    7300, 6650, 7300, 6650, 5400, 8350, 7850, 9450, 10150, 10358, 9243, 8350, 7850, 9450, 10150, 10360,
                    9242,
                    8350, 7850, 9450, 10150, 11400, 12050, 12050, 11400, 12050, 11400, 10150, 13100, 12600, 14200,
                    14900,
                    15108, 13993, 13100, 12600, 14200, 14900, 15110, 13992, 13100, 12600, 14200, 14900, 16150, 16800,
                    16800,
                    16150, 16800, 16150, 14900, 19800, 19800, 19800, 19800, 200, 200, 200]
        y_coords = [2300, 3300, 5750, 5750, 7103, 7102, 6950, 7250, 8450, 8450, 10053, 10052, 10800, 10950, 11650,
                    11650,
                    10800, 10950, 7250, 6950, 3300, 2300, 1600, 2300, 3300, 5750, 5750, 7103, 7102, 6950, 7250, 8450,
                    8450,
                    10053, 10052, 10800, 10950, 11650, 11650, 10800, 10950, 7250, 6950, 3300, 2300, 1600, 2300, 3300,
                    5750,
                    5750, 7103, 7102, 6950, 7250, 8450, 8450, 10053, 10052, 10800, 10950, 11650, 11650, 10800, 10950,
                    7250,
                    6950, 3300, 2300, 1600, 800, 10000, 11900, 12200, 12200, 1100, 800]
        #optimal_cost = 108159
    elif val == 6:
        Prob = 'Rat99'
        x_coords = [6, 15, 24, 33, 48, 57, 67, 77, 86, 6, 17, 23, 32, 43, 55, 65, 78, 87, 3, 12, 28, 33, 47, 55, 64, 71,
                    87, 4,
                    15, 22, 34, 42, 54, 66, 78, 87, 7, 17, 26, 32, 43, 57, 64, 78, 83, 5, 13, 25, 38, 46, 58, 67, 74,
                    88, 2,
                    17, 23, 36, 42, 53, 63, 72, 87, 2, 16, 25, 38, 42, 57, 66, 73, 86, 5, 13, 25, 35, 46, 54, 65, 73,
                    86, 2,
                    14, 28, 38, 46, 57, 63, 77, 85, 8, 12, 22, 34, 47, 58, 66, 78, 85]
        y_coords = [4, 15, 18, 12, 12, 14, 10, 10, 15, 21, 26, 25, 35, 23, 35, 36, 39, 35, 53, 44, 53, 49, 46, 52, 50,
                    57, 57,
                    72, 78, 70, 71, 79, 77, 79, 67, 73, 81, 95, 98, 97, 88, 89, 85, 83, 98, 109, 111, 102, 119, 107,
                    110, 110,
                    113, 110, 124, 134, 129, 131, 137, 123, 135, 134, 129, 146, 147, 153, 155, 158, 154, 151, 151, 149,
                    177,
                    162, 169, 177, 172, 166, 174, 161, 162, 195, 196, 189, 187, 195, 194, 188, 193, 194, 211, 217, 210,
                    216,
                    203, 213, 206, 210, 204]
        #optimal_cost = 1211
    elif val == 7:
        Prob = 'KroA100'
        x_coords = [1380, 2848, 3510, 457, 3888, 984, 2721, 1286, 2716, 738, 1251, 2728, 3815, 3683, 1247, 123, 1234, 252, 611,
             2576, 928, 53, 1807, 274, 2574, 178, 2678, 1795, 3384, 3520, 1256, 1424, 3913, 3085, 2573, 463, 3875, 298,
             3479, 2542, 3955, 1323, 3447, 2936, 1621, 3373, 1393, 3874, 938, 3022, 2482, 3854, 376, 2519, 2945, 953,
             2628, 2097, 890, 2139, 2421, 2290, 1115, 2588, 327, 241, 1917, 2991, 2573, 19, 3911, 872, 2863, 929, 839,
             3893, 2178, 3822, 378, 1178, 2599, 3416, 2961, 611, 3113, 2597, 2586, 161, 1429, 742, 1625, 1187, 1787, 22,
             3640, 3756, 776, 1724, 198, 3950]
        y_coords = [939, 96, 1671, 334, 666, 965, 1482, 525, 1432, 1325, 1832, 1698, 169, 1533, 1945, 862, 1946, 1240, 673,
             1676, 1700, 857, 1711, 1420, 946, 24, 1825, 962, 1498, 1079, 61, 1728, 192, 1528, 1969, 1670, 598, 1513,
             821, 236, 1743, 280, 1830, 337, 1830, 1646, 1368, 1318, 955, 474, 1183, 923, 825, 135, 1622, 268, 1479,
             981, 1846, 1806, 1007, 1810, 1052, 302, 265, 341, 687, 792, 599, 674, 1673, 1559, 558, 1766, 620, 102,
             1619, 899, 1048, 100, 901, 143, 1605, 1384, 885, 1830, 1286, 906, 134, 1025, 1651, 706, 1009, 987, 43, 882,
             392, 1642, 1810, 1558]

    #optimal_cost = 629
    elif val == 8:
        Prob = 'KroB100'
        x_coords = [3140, 556, 3675, 1182, 3595, 962, 2030, 3507, 2642, 3438, 3858, 2937, 376, 839, 706, 749, 298, 694, 387,
             2801, 3133, 1517, 1538, 844, 2639, 3123, 2489, 3834, 3417, 2938, 71, 3245, 731, 2312, 2426, 380, 2310,
             2830, 3829, 3684, 171, 627, 1490, 61, 422, 2698, 2372, 177, 3084, 1213, 3, 1782, 3896, 1829, 1286, 3017,
             2132, 2000, 3317, 1729, 2408, 3292, 193, 782, 2503, 1697, 3821, 3370, 3162, 3938, 2741, 2330, 3918, 1794,
             2929, 3453, 896, 399, 2614, 2800, 2630, 563, 1090, 2009, 3876, 3084, 1526, 1612, 1423, 3058, 3782, 347,
             3904, 2191, 3220, 468, 3611, 3114, 3515, 3060]
        y_coords = [1401, 1056, 1522, 1853, 111, 1895, 1186, 1851, 1269, 901, 1472, 1568, 1018, 1355, 1925, 920, 615, 552, 190,
             695, 1143, 266, 224, 520, 1239, 217, 1520, 1827, 1808, 543, 1323, 1828, 1741, 1270, 1851, 478, 635, 775,
             513, 445, 514, 1261, 1123, 81, 542, 1221, 127, 1390, 748, 910, 1817, 995, 742, 812, 550, 108, 1432, 1110,
             1966, 1498, 1747, 152, 1210, 1462, 352, 1924, 147, 791, 367, 516, 1583, 741, 1088, 1589, 485, 1998, 705,
             850, 195, 653, 20, 1513, 1652, 1163, 1165, 774, 1612, 328, 1322, 1276, 1865, 252, 1444, 1579, 1454, 319,
             1968, 1629, 1892, 155]

    #optimal_cost = 22141
    elif val == 9:
        Prob = 'KroC100'
        x_coords = [1357, 2650, 1774, 1307, 3806, 2687, 43, 3092, 185, 834, 40, 1183, 2048, 1097, 1838, 234, 3314, 737, 779,
             2312, 2576, 3078, 2781, 705, 3409, 323, 1660, 3729, 693, 2361, 2433, 554, 913, 3586, 2636, 1000, 482, 3704,
             3635, 1362, 2049, 2552, 3939, 219, 812, 901, 2513, 242, 826, 3278, 86, 14, 1327, 2773, 2469, 3835, 1031,
             3853, 1868, 1544, 457, 3174, 192, 2318, 2232, 396, 2365, 2499, 1410, 2990, 3646, 3394, 1779, 1058, 2933,
             3099, 2178, 138, 2082, 2302, 805, 22, 3213, 99, 1533, 3564, 29, 3808, 2221, 3499, 3124, 781, 1027, 3249,
             3297, 213, 721, 3736, 868, 960]
        y_coords = [1905, 802, 107, 964, 746, 1353, 1957, 1668, 1542, 629, 462, 1391, 1628, 643, 1732, 1118, 1881, 1285, 777,
             1949, 189, 1541, 478, 1812, 1917, 1714, 1556, 1188, 1383, 640, 1538, 1825, 317, 1909, 727, 457, 1337, 1082,
             1174, 1526, 417, 1909, 640, 898, 351, 1552, 1572, 584, 1226, 799, 1065, 454, 1893, 1286, 1838, 963, 428,
             1712, 197, 863, 1607, 1064, 1004, 1925, 1374, 828, 1649, 658, 307, 214, 1018, 1028, 90, 372, 1459, 173,
             978, 1610, 1753, 1127, 272, 1617, 1085, 536, 1780, 676, 6, 1375, 291, 1885, 408, 671, 1041, 378, 491, 220,
             186, 1542, 731, 303]

    #optimal_cost = 20749


    #optimal_cost = 22141
    else:
        raise ValueError("Invalid dataset value")

    return x_coords, y_coords


# Define Objective Function for TSP (Total Distance)
@lru_cache(maxsize=None)
def cached_distance(index1, index2):
    return np.linalg.norm([x_coords[index1] - x_coords[index2], y_coords[index1] - y_coords[index2]])

def tsp_objective_function(route):
    return sum(cached_distance(route[i], route[(i + 1) % len(route)]) for i in range(len(route)))

def initialize_tsp_population(num_hummingbirds, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(num_hummingbirds)]

def update_memory_table(memory_table, hummingbirds):
    new_memory = {}
    for route in hummingbirds:
        r_tuple = tuple(route)
        if r_tuple in memory_table:
            new_memory[r_tuple] = {'last_visited': 0, 'quality': memory_table[r_tuple]['quality']}
        else:
            new_memory[r_tuple] = {'last_visited': 0, 'quality': 1 / (tsp_objective_function(route) + 1e-10)}
    for key, value in memory_table.items():
        if key not in new_memory:
            new_memory[key] = {'last_visited': value['last_visited'] + 1, 'quality': value['quality']}
    return new_memory

def select_route(memory_table):
    return max(memory_table.items(), key=lambda x: (x[1]['quality'], -x[1]['last_visited']))[0]

def local_search_tsp(route):
    best_cost = tsp_objective_function(route)
    best_route = route[:]
    for i in range(1, len(route) - 1):
        for j in range(i + 1, len(route)):
            new_route = route[:]
            new_route[i:j] = reversed(new_route[i:j])
            new_cost = tsp_objective_function(new_route)
            if new_cost < best_cost:
                best_cost = new_cost
                best_route = new_route
    return best_route

def guided_foraging(route):
    best_route = route[:]
    best_cost = tsp_objective_function(route)
    for i in range(1, len(route) - 1):
        for j in range(i + 1, len(route)):
            new_route = route[:]
            new_route[i:j] = reversed(new_route[i:j])
            new_cost = tsp_objective_function(new_route)
            if new_cost < best_cost:
                best_cost = new_cost
                best_route = new_route
    return best_route

def apply_flight_skills(route, memory_table, flight_chance=0.3):
    if random.random() < flight_chance:
        skill_type = random.choice(['diagonal', 'axial', 'omnidirectional'])
        if skill_type == 'diagonal':
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        elif skill_type == 'axial':
            i, j = sorted(random.sample(range(len(route)), 2))
            route[i:j] = reversed(route[i:j])
        elif skill_type == 'omnidirectional':
            random.shuffle(route)
    return route





def territorial_foraging(hummingbirds, memory_table, mutation_rate=0.1):
    best_route = min(hummingbirds, key=tsp_objective_function)
    for i in range(len(hummingbirds)):
        if random.random() < 0.1:
            hummingbirds[i] = guided_foraging(hummingbirds[i])
        else:
            hummingbirds[i] = local_search_tsp(hummingbirds[i])
        current_cost = tsp_objective_function(hummingbirds[i])
        if current_cost < tsp_objective_function(best_route):
            best_route = hummingbirds[i]
    return hummingbirds, best_route

def crossover_and_mutation(parent1, parent2, mutation_rate=0.1):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    new_route = parent1[start:end]
    remaining_cities = [city for city in parent2 if city not in new_route]
    new_route = remaining_cities[:start] + new_route + remaining_cities[start:]
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(size), 2)
        new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
    return new_route

# AHA Algorithm for TSP
def aha_algorithm_tsp(max_iterations=1000, num_hummingbirds=20, flight_chance=0.3, mutation_rate=0.1):
    num_cities = len(x_coords)
    hummingbirds = initialize_tsp_population(num_hummingbirds, num_cities)
    memory_table = {tuple(h): {'last_visited': 0, 'quality': 1 / (tsp_objective_function(h) + 1e-10)} for h in hummingbirds}

    # Plot initialization
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o-', lw=2)
    title = ax.text(0.5, 0.95, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                    transform=ax.transAxes, ha="center")

    def init():
        ax.set_xlim(min(x_coords) - 10, max(x_coords) + 10)
        ax.set_ylim(min(y_coords) - 10, max(y_coords) + 10)
        return line, title

    # Update function for animation
    def update(iteration):
        nonlocal memory_table, hummingbirds
        for i in range(len(hummingbirds)):
            if random.random() < flight_chance:
                hummingbirds[i] = apply_flight_skills(hummingbirds[i], memory_table)
                parent1, parent2 = random.sample(hummingbirds, 2)
                hummingbirds[i] = crossover_and_mutation(parent1, parent2, mutation_rate)
            hummingbirds[i] = local_search_tsp(hummingbirds[i])
        memory_table = update_memory_table(memory_table, hummingbirds)
        hummingbirds, best_route = territorial_foraging(hummingbirds, memory_table)
        best_route_tuple = select_route(memory_table)
        best_route = list(best_route_tuple)
        best_cost = tsp_objective_function(best_route)
        print(f"Iteration {iteration + 1}: Best Cost = {best_cost:.2f}")
        line.set_data([x_coords[i] for i in best_route + [best_route[0]]],
                      [y_coords[i] for i in best_route + [best_route[0]]])
        title.set_text(f"Iteration {iteration}: Best Cost = {best_cost:.2f}")
        return line, title

    ani = FuncAnimation(fig, update, frames=np.arange(0, max_iterations), init_func=init, blit=True, repeat=False)
    plt.show()

    best_route_tuple = select_route(memory_table)
    best_route = list(best_route_tuple)
    best_cost = tsp_objective_function(best_route)
    return best_route, best_cost

# Function to run the algorithm multiple times and collect statistics
def run_multiple_executions(num_executions=1):
    costs = []
    BKSs = []

    for _ in range(num_executions):
        best_route, best_cost = aha_algorithm_tsp()
        costs.append(best_cost)

    best_cost = min(costs)
    worst_cost = max(costs)
    avg_cost = np.mean(costs)



    print(f"Meilleur coût: {best_cost}")
    print(f"Pire coût: {worst_cost}")
    print(f"Moyenne des coûts: {avg_cost}")


print("Sélectionnez votre problème - insérez un nombre de 1 à 6 -")
print(" 1. Eil51")
print(" 2. Berlin52")
print(" 3. St70")
print(" 4. Eil76")
print(" 5. Pr76")
print(" 6. Rat99")
print("7. KroA100 ")
print("8. KroB100 ")

print("9. kroc100")


val = input("Veuillez insérer votre numéro de 1 à 10: ")
try:
    val = int(val)
    if val < 1 or val > 10:
        raise ValueError("La valeur doit être comprise entre 1 et 6")
except ValueError:
    print("Entrée invalide. Veuillez entrer un numéro valide entre 1 et 6.")
    exit()

# Load dataset based on the selected value
x_coords, y_coords = load_dataset(val)

# Run the algorithm multiple times and print the results
run_multiple_executions()
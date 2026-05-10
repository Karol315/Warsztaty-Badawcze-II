import numpy as np
import heapq

def heuristic(a, b):
    # Dystans Manhattan
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid: np.ndarray, start: tuple, goal: tuple):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    # --- NOWOŚĆ: Śledzimy, jak daleko udało nam się dojść ---
    closest_node = start
    min_h = heuristic(start, goal)

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            # Znalazł pełną trasę do samego wyjścia!
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        close_set.add(current)
        
        # Jeśli ten punkt jest bliżej celu niż cokolwiek wcześniej, zapamiętaj go
        current_h = heuristic(current, goal)
        if current_h < min_h:
            min_h = current_h
            closest_node = current

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue

            tentative_g_score = gscore[current] + 1
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    # --- ZMIANA: Jeśli droga do celu jest odcięta (brak rozwiązania), 
    # zwracamy trasę do punktu, do którego w ogóle udało nam się zbliżyć najbardziej. ---
    path = []
    current = closest_node
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1]
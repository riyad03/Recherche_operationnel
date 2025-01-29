
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk 
from tkinter import ttk, simpledialog,messagebox,Tk
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import string
import pandas as pd
import numpy as np
from tabulate import tabulate
import os

# Constants for styling
EMSI_GREEN = "#006838"
DARK_GRAY = "#333333"
WINDOW_BG = "#FFFFFF"

# Function to show graph in 2D in a new window (with Tkinter integration)
def show_graph_in_new_window_2d(graph, title, path=None, mst_edges=None, bellman_ford_paths=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    pos = nx.spring_layout(graph)
    node_colors = ['lightblue' if node not in path else 'green' for node in graph.nodes()] if path else 'lightblue'
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=700, ax=ax)

    edge_colors = []
    for edge in graph.edges():
        edge_color = 'gray'
        if mst_edges and edge in mst_edges:
            edge_color = 'red'
        elif path and (edge[0] in path and edge[1] in path):
            edge_color = 'red'
        elif bellman_ford_paths:
            for p in bellman_ford_paths.values():
                if edge[0] in p and edge[1] in p and p.index(edge[1]) == p.index(edge[0]) + 1:
                    edge_color = 'red'
                    break
        edge_colors.append(edge_color)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, ax=ax)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)

    nx.draw_networkx_labels(graph, pos, ax=ax)
    ax.set_title(title)

    # Creating a Tkinter window to embed the plot
    plot_window = tk.Toplevel()
    plot_window.title(title)

    # Embed the matplotlib plot into Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Welsh-Powell algorithm
def generate_random_graph(num_vertices, probability):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < probability:
                graph.add_edge(i, j)
    return graph

def welsh_powell2(graph):
    # Step 1: Sort nodes by degree in decreasing order
    sorted_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    sorted_nodes = [node for node, degree in sorted_nodes]
    
    # Step 2: Initialize colors dictionary
    colors = {}
    
    # Step 3: Start with the first node, assigning the first color
    available_colors = list(range(len(graph.nodes())))
    used_colors = {}
    
    for node in sorted_nodes:
        if node not in colors:
            # Step 4: Find the lowest available color for the current node
            neighbor_colors = {colors[neighbor] for neighbor in graph.neighbors(node) if neighbor in colors}
            for color in available_colors:
                if color not in neighbor_colors:
                    colors[node] = color
                    used_colors[color] = True
                    break
            # Step 5: Remove the used color from available colors
            available_colors = [color for color in available_colors if color not in neighbor_colors]
    
    return colors

def generate_random_connected_graph(num_nodes, num_edges):
    """Generate a connected graph with random edges without making it fully connected."""
    if num_edges < num_nodes - 1:
        raise ValueError("Number of edges must be at least n-1 to ensure connectivity.")

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    # Step 1: Create a spanning tree (ensures all nodes are connected)
    nodes = list(range(num_nodes))
    random.shuffle(nodes)

    for i in range(num_nodes - 1):
        graph.add_edge(nodes[i], nodes[i + 1])

    # Step 2: Add additional random edges (without making it fully connected)
    extra_edges = num_edges - (num_nodes - 1)
    while extra_edges > 0:
        u, v = random.sample(range(num_nodes), 2)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
            extra_edges -= 1

    return graph

def welsh_powell_coloring(graph):
    sorted_nodes = sorted(graph.nodes, key=lambda x: graph.degree[x], reverse=True)
    colors = {}
    current_color = 0

    for node in sorted_nodes:
        if node not in colors:
            current_color += 1
            colors[node] = current_color
            for other_node in sorted_nodes:
                if other_node not in colors and all(colors.get(neighbor) != current_color for neighbor in graph.neighbors(other_node)):
                    colors[other_node] = current_color
    return colors, current_color

def execute_welsh_powell_algorithm():
    try:
        num_nodes = int(simpledialog.askstring("Entrée", "Entrez le nombre de sommets du graphe :"))
        num_edges = num_nodes-1
    except (ValueError, TypeError):
        messagebox.showerror("Erreur", "Entrées invalides. Veuillez entrer des entiers valides.")
        return

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    edges = set()
    while len(edges) < num_edges:
        u, v = random.sample(range(num_nodes), 2)
        if u != v and (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
    graph.add_edges_from(edges)

    colors, chromatic_number = welsh_powell_coloring(graph)

    result = f"Nombre chromatique (Chromatic Number) : {chromatic_number}\n"
    result += "\n".join([f"Sommets {node} -> Couleur {color}" for node, color in colors.items()])
    messagebox.showinfo("Résultats Welsh-Powell", result)

    pos = nx.spring_layout(graph)
    color_map = [colors[node] for node in graph.nodes]
    plt.figure(figsize=(8, 6))
    nx.draw(
        graph, pos, with_labels=True, node_color=color_map, 
        node_size=500, cmap=plt.cm.rainbow, font_color='white'
    )
    plt.title(f"Coloriage du graphe (Nombre chromatique : {chromatic_number})")
    plt.show()


# Dijkstra algorithm
def generate_weighted_graph(num_vertices, probability):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < probability:
                weight = random.randint(1, 100)
                graph.add_edge(i, j, weight=weight)
    return graph

def dijkstra(graph, start, end):
    path = nx.dijkstra_path(graph, start, end, weight='weight')
    return path

# Kruskal algorithm
def generate_labeled_weighted_graph(num_vertices, probability):
    graph = nx.Graph()
    labels = [c for c in string.ascii_uppercase[:num_vertices]]
    graph.add_nodes_from(labels)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < probability:
                weight = random.randint(1, 100)
                graph.add_edge(labels[i], labels[j], weight=weight)
    return graph

def kruskal(graph):
    edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])
    mst = nx.Graph()
    mst.add_nodes_from(graph.nodes())
    union_find = {node: node for node in graph.nodes()}

    def find(node):
        if union_find[node] != node:
            union_find[node] = find(union_find[node])
        return union_find[node]

    def union(node1, node2):
        root1, root2 = find(node1), find(node2)
        union_find[root1] = root2

    for edge in edges:
        node1, node2, data = edge
        if find(node1) != find(node2):
            mst.add_edge(node1, node2, weight=data['weight'])
            union(node1, node2)

    return mst

# Bellman-Ford algorithm
def generate_random_weighted_digraph(num_nodes, prob_edge=0.3, min_weight=0, max_weight=10):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u != v and random.random() < prob_edge:
                weight = random.randint(min_weight, max_weight)
                graph.add_edge(u, v, weight=weight)
    
    return graph

def bellman_ford(graph, source):
    try:
        shortest_paths = nx.single_source_bellman_ford_path(graph, source)
        shortest_distances = nx.single_source_bellman_ford_path_length(graph, source)
        return shortest_paths, shortest_distances
    except nx.NetworkXUnbounded:
        return None, None

# Potentiel-Metra algorithm
def generer_taches(nb_taches):
    taches = []
    for i in range(nb_taches):
        duree = random.randint(1, 10)
        jour_debut = random.randint(1, 30)
        taches.append({
            'Tache': f'Tâche {i+1}',
            'Durée': duree,
            'Jour Début': jour_debut
        })
    return taches

def appliquer_methode_potentiel(taches):
    taches_sorted = sorted(taches, key=lambda x: x['Jour Début'])
    
    for tache in taches_sorted:
        tache['Jour Fin'] = tache['Jour Début'] + tache['Durée'] - 1
    
    for i, tache in enumerate(taches_sorted):
        if i == 0:
            tache['Marge Plus Tôt'] = 0
            tache['Marge Plus Tard'] = tache['Jour Fin']
        else:
            prev_tache = taches_sorted[i - 1]
            tache['Marge Plus Tôt'] = prev_tache['Jour Fin']
            tache['Marge Plus Tard'] = tache['Jour Fin'] + tache['Durée'] - 1

    return taches_sorted

# Ford-Fulkerson algorithm
def generate_flow_network(num_vertices, max_capacity=10):
    G = nx.DiGraph()
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j:
                capacity = random.randint(1, max_capacity)
                G.add_edge(i, j, capacity=capacity)
    return G

def bfs(capacity, flow, source, sink):
    parent = [-1] * len(capacity)
    parent[source] = -2
    queue = deque([(source, float('inf'))])
    while queue:
        u, min_cap = queue.popleft()
        
        for v in range(len(capacity)):
            if parent[v] == -1 and capacity[u][v] - flow[u][v] > 0:
                parent[v] = u
                new_flow = min(min_cap, capacity[u][v] - flow[u][v])
                if v == sink:
                    return new_flow, parent
                queue.append((v, new_flow))
    return 0, parent

def ford_fulkerson(capacity, source, sink):
    n = len(capacity)
    flow = [[0] * n for _ in range(n)]
    max_flow = 0
    
    while True:
        path_flow, parent = bfs(capacity, flow, source, sink)
        if path_flow == 0:
            break
        max_flow += path_flow
        
        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            v = u
    return max_flow, flow

def find_min_cut(capacity, flow, source):
    visited = [False] * len(capacity)
    queue = deque([source])
    visited[source] = True
    
    while queue:
        u = queue.popleft()
        for v in range(len(capacity)):
            if capacity[u][v] - flow[u][v] > 0 and not visited[v]:
                visited[v] = True



    return visited

# Stepping Stone algorithm
def generate_data(nb_usines, nb_magasins, min_cost=1, max_cost=20, min_cap=10, max_cap=50):
    couts = np.random.randint(min_cost, max_cost, size=(nb_usines, nb_magasins))
    capacites = np.random.randint(min_cap, max_cap, size=nb_usines)
    demandes = np.random.randint(min_cap, max_cap, size=nb_magasins)

    total_capacite = sum(capacites)
    total_demande = sum(demandes)
    if total_capacite > total_demande:
        demandes[-1] += total_capacite - total_demande
    else:
        capacites[-1] += total_demande - total_capacite

    return couts, capacites, demandes

def calculer_cout_total(couts, allocation):
    return np.sum(couts * allocation)

def nord_ouest(capacites, demandes):
    allocation = np.zeros((len(capacites), len(demandes)), dtype=int)
    i, j = 0, 0
    while i < len(capacites) and j < len(demandes):
        alloc = min(capacites[i], demandes[j])
        allocation[i, j] = alloc
        capacites[i] -= alloc
        demandes[j] -= alloc
        if capacites[i] == 0:
            i += 1
        if demandes[j] == 0:
            j += 1
    return allocation

def moindre_cout(couts, capacites, demandes):
    allocation = np.zeros_like(couts, dtype=int)
    couts_temp = couts.astype(float)
    while np.any(capacites) and np.any(demandes):
        i, j = np.unravel_index(np.argmin(couts_temp, axis=None), couts_temp.shape)
        alloc = min(capacites[i], demandes[j])
        allocation[i, j] = alloc
        capacites[i] -= alloc
        demandes[j] -= alloc
        if capacites[i] == 0:
            couts_temp[i, :] = np.inf
        if demandes[j] == 0:
            couts_temp[:, j] = np.inf
    return allocation

def stepping_stone(couts, allocation):
    rows, cols = allocation.shape
    couts = couts.astype(float)
    while True:
        empty_cells = [(i, j) for i in range(rows) for j in range(cols) if allocation[i, j] == 0]
        best_improvement = 0
        best_allocation = allocation.copy()
        
        for cell in empty_cells:
            cycle, gain = find_cycle_and_gain(couts, allocation, cell)
            if cycle and gain < best_improvement:
                best_improvement = gain
                best_allocation = adjust_allocation(allocation, cycle)

        if best_improvement >= 0:
            break
        allocation = best_allocation

    return allocation

def find_cycle_and_gain(couts, allocation, start_cell):
    rows, cols = allocation.shape
    visited = set()
    cycle = []

    def dfs(cell, path):
        if cell in visited:
            if cell == start_cell and len(path) >= 4:
                return path
            return None

        visited.add(cell)
        row, col = cell

        for next_cell in [(row, c) for c in range(cols)] + [(r, col) for r in range(rows)]:
            if next_cell != cell and allocation[next_cell] > 0 or next_cell == start_cell:
                new_path = dfs(next_cell, path + [cell])
                if new_path:
                    return new_path

        visited.remove(cell)
        return None

    cycle = dfs(start_cell, [])

    if not cycle:
        return None, 0

    gain = calculate_cycle_gain(couts, allocation, cycle)
    return cycle, gain

def calculate_cycle_gain(couts, allocation, cycle):
    gain = 0
    for k, (i, j) in enumerate(cycle):
        sign = 1 if k % 2 == 0 else -1
        gain += sign * couts[i, j]
    return gain

def adjust_allocation(allocation, cycle):
    min_alloc = min(allocation[i, j] for k, (i, j) in enumerate(cycle) if k % 2 == 1)

    for k, (i, j) in enumerate(cycle):
        sign = 1 if k % 2 == 0 else -1
        allocation[i, j] += sign * min_alloc

    return allocation

def afficher_tableau(data, row_labels=None, col_labels=None, title=None):
    if row_labels is not None and col_labels is not None:
        table = tabulate(data, headers=col_labels, showindex=row_labels, tablefmt="fancy_grid")
    else:
        table = tabulate(data, tablefmt="fancy_grid")
    if title:
        print(f"\n{title}\n{'=' * len(title)}")
    print(table)

# Main interface functions
class ModernButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            relief=tk.FLAT,
            bg=EMSI_GREEN,
            fg="white",
            font=("Helvetica", 11),
            cursor="hand2",
            pady=8
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['background'] = DARK_GRAY

    def on_leave(self, e):
        self['background'] = EMSI_GREEN

def create_modern_window(title, geometry):
    window = tk.Toplevel()
    window.title(title)
    window.geometry(geometry)
    window.configure(bg=WINDOW_BG)
    return window

def execute_welsh_powell_algorithm2():
    window = create_modern_window("Welsh-Powell", "400x300")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)

    '''tk.Label(main_frame, text="Probabilité (0-1) :", bg=WINDOW_BG).pack(pady=5)
    probability_entry = tk.Entry(main_frame)
    probability_entry.pack(pady=5)'''

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            probability = random.uniform(0.75, 1)
            graph = generate_random_graph(num_vertices, probability)
            colors = welsh_powell(graph)
            chromatic_number = len(set(colors.values()))
            result_label.config(text=f"Nombre chromatique : {chromatic_number}")
            show_graph_in_new_window_2d(graph, "Welsh-Powell Graph")
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)



def execute_dijkstra_algorithm():
    window = create_modern_window("Dijkstra", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)

    '''tk.Label(main_frame, text="Probabilité (0-1) :", bg=WINDOW_BG).pack(pady=5)
    probability_entry = tk.Entry(main_frame)
    probability_entry.pack(pady=5)'''

    tk.Label(main_frame, text="Sommet de départ :", bg=WINDOW_BG).pack(pady=5)
    start_entry = tk.Entry(main_frame)
    start_entry.pack(pady=5)

    tk.Label(main_frame, text="Sommet d'arrivée :", bg=WINDOW_BG).pack(pady=5)
    end_entry = tk.Entry(main_frame)
    end_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            probability = random.uniform(0.75, 1)
            start = int(start_entry.get())
            end = int(end_entry.get())
            graph = generate_weighted_graph(num_vertices, probability)
            path = dijkstra(graph, start, end)
            result_label.config(text=f"Chemin le plus court : {' -> '.join(map(str, path))}")
            show_graph_in_new_window_2d(graph, "Graphe Dijkstra", path)
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_kruskal_algorithm():
    window = create_modern_window("Kruskal", "400x300")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    
    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)
    '''
    tk.Label(main_frame, text="Probabilité (0-1) :", bg=WINDOW_BG).pack(pady=5)
    probability_entry = tk.Entry(main_frame)
    probability_entry.pack(pady=5)'''

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            probability = random.uniform(0.75, 1)
            graph = generate_labeled_weighted_graph(num_vertices, probability)
            mst = kruskal(graph)
            total_weight = sum(mst[u][v]['weight'] for u, v in mst.edges())
            result_label.config(text=f"Poids total de l'arbre couvrant minimal : {total_weight}")
            show_graph_in_new_window_2d(graph, "Graphe Kruskal", mst_edges=mst.edges())
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_bellman_ford_algorithm():
    window = create_modern_window("Bellman-Ford", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)
    '''
    tk.Label(main_frame, text="Probabilité d'arête (0-1) :", bg=WINDOW_BG).pack(pady=5)
    probability_entry = tk.Entry(main_frame)
    probability_entry.pack(pady=5)'''

    tk.Label(main_frame, text="Sommet source :", bg=WINDOW_BG).pack(pady=5)
    source_entry = tk.Entry(main_frame)
    source_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            probability = random.uniform(0.75, 1)
            source = int(source_entry.get())
            graph = generate_random_weighted_digraph(num_vertices, probability)
            shortest_paths, shortest_distances = bellman_ford(graph, source)
            
            if shortest_paths is None:
                result_label.config(text="Le graphe contient un cycle de poids négatif.")
            else:
                result_text = f"Résultats de Bellman-Ford depuis le sommet {source}:\n"
                for target, path in shortest_paths.items():
                    distance = shortest_distances[target]
                    result_text += f"Vers {target}: {' -> '.join(map(str, path))} (distance: {distance})\n"
                result_label.config(text=result_text)
                show_graph_in_new_window_2d(graph, "Graphe Bellman-Ford", bellman_ford_paths=shortest_paths)
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_potentiel_metra_algorithm():
    window = create_modern_window("Potentiel-Metra", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de tâches :", bg=WINDOW_BG).pack(pady=5)
    tasks_entry = tk.Entry(main_frame)
    tasks_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            nb_taches = int(tasks_entry.get())
            taches = generer_taches(nb_taches)
            taches_calculees = appliquer_methode_potentiel(taches)
            
            df = pd.DataFrame(taches_calculees)
            result_text = "Tableau des tâches avec dates de début, fin et marges :\n"
            result_text += df.to_string(index=False)
            result_label.config(text=result_text)

            # Create a Gantt chart
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, task in enumerate(taches_calculees):
                ax.barh(i, task['Durée'], left=task['Jour Début'], height=0.5)
                ax.text(task['Jour Début'], i, task['Tache'], va='center', ha='right', fontweight='bold')
            
            ax.set_yticks(range(len(taches_calculees)))
            ax.set_yticklabels([task['Tache'] for task in taches_calculees])
            ax.set_xlabel('Jours')
            ax.set_title('Diagramme de Gantt des tâches')
            
            # Show the Gantt chart in a new window
            chart_window = tk.Toplevel()
            chart_window.title("Diagramme de Gantt")
            canvas = FigureCanvasTkAgg(fig, master=chart_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_ford_fulkerson_algorithm():
    window = create_modern_window("Ford-Fulkerson", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)

    tk.Label(main_frame, text="Capacité maximale :", bg=WINDOW_BG).pack(pady=5)
    max_capacity_entry = tk.Entry(main_frame)
    max_capacity_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            max_capacity = int(max_capacity_entry.get())
            graph = generate_flow_network(num_vertices, max_capacity)
            
            source = 0
            sink = num_vertices - 1
            
            capacity = [[0] * num_vertices for _ in range(num_vertices)]
            for u, v, data in graph.edges(data=True):
                capacity[u][v] = data['capacity']
            
            max_flow, flow = ford_fulkerson(capacity, source, sink)
            
            result_text = f"Flot maximal : {max_flow}\n\n"
            result_text += "Flots sur les arcs :\n"
            for u in range(num_vertices):
                for v in range(num_vertices):
                    if flow[u][v] > 0:
                        result_text += f"De {u} à {v}: {flow[u][v]}/{capacity[u][v]}\n"
            
            result_label.config(text=result_text)
            
            # Visualize the flow network
            pos = nx.spring_layout(graph)
            plt.figure(figsize=(10, 8))
            nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
            
            edge_labels = {(u, v): f"{flow[u][v]}/{data['capacity']}" for u, v, data in graph.edges(data=True)}
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
            
            plt.title("Réseau de flot avec flots/capacités")
            
            # Show the flow network in a new window
            chart_window = tk.Toplevel()
            chart_window.title("Réseau de flot")
            canvas = FigureCanvasTkAgg(plt.gcf(), master=chart_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_stepping_stone_algorithm():
    window = create_modern_window("Stepping Stone", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre d'usines :", bg=WINDOW_BG).pack(pady=5)
    nb_usines_entry = tk.Entry(main_frame)
    nb_usines_entry.pack(pady=5)

    tk.Label(main_frame, text="Nombre de magasins :", bg=WINDOW_BG).pack(pady=5)
    nb_magasins_entry = tk.Entry(main_frame)
    nb_magasins_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            nb_usines = int(nb_usines_entry.get())
            nb_magasins = int(nb_magasins_entry.get())
            
            couts, capacites, demandes = generate_data(nb_usines, nb_magasins)
            
            # Nord-Ouest
            allocation_nord_ouest = nord_ouest(capacites.copy(), demandes.copy())
            cout_nord_ouest = calculer_cout_total(couts, allocation_nord_ouest)
            
            # Moindres Coûts
            allocation_moindre_cout = moindre_cout(couts, capacites.copy(), demandes.copy())
            cout_moindre_cout = calculer_cout_total(couts, allocation_moindre_cout)
            
            # Stepping Stone
            allocation_optimisee = stepping_stone(couts, allocation_moindre_cout)
            cout_optimise = calculer_cout_total(couts, allocation_optimisee)
            
            result_text = f"Coût total (Nord-Ouest): {cout_nord_ouest}\n"
            result_text += f"Coût total (Moindres Coûts): {cout_moindre_cout}\n"
            result_text += f"Coût total optimisé (Stepping Stone): {cout_optimise}\n\n"
            
            result_text += "Allocation optimisée:\n"
            result_text += tabulate(allocation_optimisee, 
                                    headers=[f"Magasin {j+1}" for j in range(nb_magasins)],
                                    showindex=[f"Usine {i+1}" for i in range(nb_usines)])
            
            result_label.config(text=result_text)
            
            # Visualize the optimized allocation
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(allocation_optimisee, cmap='YlOrRd')
            
            ax.set_xticks(np.arange(nb_magasins))
            ax.set_yticks(np.arange(nb_usines))
            ax.set_xticklabels([f"M{j+1}" for j in range(nb_magasins)])
            ax.set_yticklabels([f"U{i+1}" for i in range(nb_usines)])
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            for i in range(nb_usines):
                for j in range(nb_magasins):
                    text = ax.text(j, i, allocation_optimisee[i, j], ha="center", va="center", color="black")
            
            ax.set_title("Allocation optimisée")
            fig.tight_layout()
            
            # Show the allocation visualization in a new window
            chart_window = tk.Toplevel()
            chart_window.title("Allocation optimisée")
            canvas = FigureCanvasTkAgg(fig, master=chart_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)



def execute_stepping_stone_algorithm2():
    window = create_modern_window("Stepping Stone", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre d'usines :", bg=WINDOW_BG).pack(pady=5)
    nb_usines_entry = tk.Entry(main_frame)
    nb_usines_entry.pack(pady=5)

    tk.Label(main_frame, text="Nombre de magasins :", bg=WINDOW_BG).pack(pady=5)
    nb_magasins_entry = tk.Entry(main_frame)
    nb_magasins_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG, justify="left")
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            nb_usines = int(nb_usines_entry.get())
            nb_magasins = int(nb_magasins_entry.get())
            
            if nb_usines <= 0 or nb_magasins <= 0:
                raise ValueError("Les nombres doivent être supérieurs à 0.")
            
            # Generate random data
            couts, capacites, demandes = generate_data(nb_usines, nb_magasins)
            
            # North-West Corner
            allocation_nord_ouest = nord_ouest(capacites, demandes)
            cout_nord_ouest = calculer_cout_total(couts, allocation_nord_ouest)
            
            # Least Cost
            allocation_moindre_cout = moindre_cout(couts, capacites, demandes)
            cout_moindre_cout = calculer_cout_total(couts, allocation_moindre_cout)
            
            # Stepping Stone Optimization
            allocation_optimisee = stepping_stone(couts, allocation_moindre_cout)
            cout_optimise = calculer_cout_total(couts, allocation_optimisee)
            
            # Display Results
            result_text = (
                f"Coût total (Nord-Ouest) : {cout_nord_ouest}\n"
                f"Coût total (Moindres Coûts) : {cout_moindre_cout}\n"
                f"Coût total optimisé (Stepping Stone) : {cout_optimise}\n\n"
                f"Allocation optimisée :\n"
                + tabulate(
                    allocation_optimisee, 
                    headers=[f"M{j+1}" for j in range(nb_magasins)], 
                    showindex=[f"U{i+1}" for i in range(nb_usines)]
                )
            )

            result_label.config(text=result_text)
            
            # Visualize Allocation
            def visualize_allocation():
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(allocation_optimisee, cmap="Blues")
                
                # Tick labels
                ax.set_xticks(range(nb_magasins))
                ax.set_yticks(range(nb_usines))
                ax.set_xticklabels([f"M{j+1}" for j in range(nb_magasins)])
                ax.set_yticklabels([f"U{i+1}" for i in range(nb_usines)])
                
                # Annotate cells
                for i in range(nb_usines):
                    for j in range(nb_magasins):
                        ax.text(j, i, allocation_optimisee[i, j], ha="center", va="center", color="black")
                
                ax.set_title("Allocation Optimisée - Stepping Stone")
                fig.colorbar(im, ax=ax)
                plt.tight_layout()
                plt.show()

            visualize_allocation()
        
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)


# Potentiel-Metra algorithm execution
def execute_potentiel_metra_algorithm2():
    # Request number of tasks
    num_tasks = ask_number_of_tasks()
    
    if not num_tasks:
        return  # Exit if the user cancels

    # Generate tasks and relations
    tasks, edges = generate_tasks_and_dependencies(num_tasks)
    
    # Calculate times
    earliest_start, latest_start, total_duration = apply_potential_method(tasks)
    critical_path = find_critical_path(earliest_start, latest_start)
    
    # Visualize the graph
    visualize_graph(tasks, edges, earliest_start, latest_start, critical_path, total_duration)

# Function to request number of tasks
def ask_number_of_tasks():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    return simpledialog.askinteger("Nombre de tâches", "Combien de tâches souhaitez-vous ?", minvalue=1, maxvalue=20)

# Function to generate tasks and dependencies
def generate_tasks_and_dependencies(num_tasks):
    tasks = {'debut': {'duration': 0, 'precedents': []}}
    edges = []

    for i in range(num_tasks):
        task_name = chr(65 + i)
        duration = random.randint(1, 5)
        precedents = random.sample(list(tasks.keys()), k=random.randint(1, min(3, len(tasks))))
        tasks[task_name] = {'duration': duration, 'precedents': precedents}

        for precedent in precedents:
            edges.append((precedent, task_name))

    final_tasks = [task for task in tasks.keys() if task not in [edge[0] for edge in edges]]
    tasks['fin'] = {'duration': 0, 'precedents': final_tasks}
    for task in final_tasks:
        edges.append((task, 'fin'))

    return tasks, edges

# Function to apply the potential method
def apply_potential_method(tasks):
    earliest_start = {}
    latest_start = {}
    total_duration = 0

    # Forward pass
    for task, attrs in tasks.items():
        if not attrs['precedents']:
            earliest_start[task] = 0
        else:
            earliest_start[task] = max(earliest_start[prec] + tasks[prec]['duration'] for prec in attrs['precedents'])

    total_duration = max(earliest_start[task] + attrs['duration'] for task, attrs in tasks.items())

    # Backward pass
    for task in reversed(list(tasks.keys())):
        if task == 'fin':
            latest_start[task] = total_duration - tasks[task]['duration']
        else:
            successors = [succ for succ, attrs in tasks.items() if task in attrs['precedents']]
            latest_start[task] = min(latest_start[succ] - tasks[task]['duration'] for succ in successors)

    return earliest_start, latest_start, total_duration

# Function to find the critical path
def find_critical_path(earliest_start, latest_start):
    return [task for task in earliest_start if earliest_start[task] == latest_start[task]]

# Function to visualize the graph
def visualize_graph(tasks, edges, earliest_start, latest_start, critical_path, total_duration):
    G = nx.DiGraph()
    
    for task, attrs in tasks.items():
        G.add_node(task, duration=attrs['duration'])
    for u, v in edges:
        G.add_edge(u, v)
    
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    
    # Node Colors
    node_colors = [
        "#ffffff" if node in critical_path else  # White for critical path
        "#b3e5fc" if node == 'debut' else      # Light blue for 'debut'
        "#c8e6c9" if node == 'fin' else        # Light green for 'fin'
        "#f5f5f5"                              # Light gray for others
        for node in G.nodes()
    ]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, edgecolors="#424242", node_shape='s')
    
    edge_colors = ["#ff0000" if u in critical_path and v in critical_path else "#b0bec5" for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowstyle='-|>', arrowsize=20)
    
    labels = {node: f"{node}\nES: {earliest_start.get(node, 'N/A')}\nLS: {latest_start.get(node, 'N/A')}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold", font_family="Georgia", font_color="#424242")
    
    plt.title("Graphe MPM - Chemin critique en rouge", fontsize=16, fontweight="bold", fontname="Georgia", color="#ff5722")
    plt.text(-1, -1.5, f"Chemin critique : {' -> '.join(critical_path)}\nDurée totale : {total_duration}", 
             fontsize=12, fontweight="bold", color="#424242")
    
    plt.axis("off")
    plt.show()



def show_second_interface():
    window = create_modern_window("Algorithmes de Graphes", "600x400")
    
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    algorithms = [
        ("Welsh-Powell", execute_welsh_powell_algorithm),
        ("Dijkstra", execute_dijkstra_algorithm),
        ("Kruskal", execute_kruskal_algorithm),
        ("Bellman-Ford", execute_bellman_ford_algorithm),
        ("Potentiel-Metra", execute_potentiel_metra_algorithm2),
        ("Ford-Fulkerson", execute_ford_fulkerson_algorithm),
        ("Stepping Stone", execute_stepping_stone_algorithm),
    ]

    for i, (text, command) in enumerate(algorithms):
        btn = ModernButton(main_frame, text=text, command=command, width=20)
        btn.grid(row=i // 3, column=i % 3, padx=10, pady=10)

# Main window setup
root = tk.Tk()
root.title("EMSI - Algorithmes de Graphes")
root.geometry("800x600")  # Increased window size for better layout
root.configure(bg=WINDOW_BG)

# Main content frame
main_frame = tk.Frame(root, bg=WINDOW_BG, padx=40, pady=30)
main_frame.pack(fill=tk.BOTH, expand=True)

# Logo frame for better centering
logo_frame = tk.Frame(main_frame, bg=WINDOW_BG)
logo_frame.pack(fill=tk.X, pady=(0, 30))

# Logo handling
image_path = "logo emsi.png"
if os.path.exists(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((300, 60))  # Slightly larger logo
        photo = ImageTk.PhotoImage(img)
        
        root.photo = photo
        image_label = tk.Label(logo_frame, image=photo, bg=WINDOW_BG)
        image_label.pack(anchor="center")  # Center the logo
    except Exception as e:
        print(f"Error loading image: {e}")

# Credits frame
credits_frame = tk.Frame(root, bg=WINDOW_BG)
credits_frame.place(relx=1, rely=1, anchor="se", x=-10, y=-10)  # Better positioning

label1 = ttk.Label(credits_frame, text="Réalisé par Mr Nassour Ismail et Mr Rachidi Riyad")
label1.pack(anchor="e")

label2 = ttk.Label(credits_frame, text="Encadrée par Mme Mouna El Mkhalet")
label2.pack(anchor="e", pady=(5, 0))

# Content
title_label = tk.Label(
    main_frame,
    text="Interface graphique des algorithmes\nde la recherche opérationnelle",  # Split into two lines
    font=("Helvetica", 22, "bold"),
    fg=EMSI_GREEN,
    bg=WINDOW_BG,
    justify=tk.CENTER
)
title_label.pack(pady=(0, 20))

subtitle_label = tk.Label(
    main_frame,
    text="Réalise par Riyad Rachidi et Ismail Nassour, Encadrée par Mme Mouna El Mkhalet",
    font=("Helvetica", 14),
    fg=DARK_GRAY,
    bg=WINDOW_BG
)
subtitle_label.pack(pady=(0, 40))

# Buttons frame
buttons_frame = tk.Frame(main_frame, bg=WINDOW_BG)
buttons_frame.pack(expand=True)  # Center vertically

start_button = ModernButton(buttons_frame, text="Commencer", width=25, command=show_second_interface)
start_button.pack()

# Footer
footer_label = tk.Label(
    main_frame,
    text="© 2024 EMSI - Tous droits réservés",
    font=("Helvetica", 9),
    fg=DARK_GRAY,
    bg=WINDOW_BG
)
footer_label.pack(side=tk.BOTTOM, pady=(0, 10))

root.mainloop()
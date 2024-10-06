import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

# Contexte 3D avec les fruits
context_3D = {
    'objects': ['Kiwi', 'Ananas', 'Cerise', 'Pamplemousse'],
    'attributes': ['Tropical', 'Agrume', 'Rouge', 'Sucré', 'Acide', 'Peut être cuisiné', 'Saison', 'Riche en vitamine C', 'Utilisé dans les desserts'],
    'seasons': ['Été', 'Hiver'],
    'relations': {
        'Été': [
            ['Kiwi', 'Tropical', 1],
            ['Kiwi', 'Sucré', 1],
            ['Kiwi', 'Acide', 1],
            ['Kiwi', 'Peut être cuisiné', 1],
            ['Kiwi', 'Saison', 1],
            ['Kiwi', 'Riche en vitamine C', 1],
            ['Kiwi', 'Utilisé dans les desserts', 1],
            ['Ananas', 'Tropical', 1],
            ['Ananas', 'Sucré', 1],
            ['Ananas', 'Acide', 1],
            ['Ananas', 'Peut être cuisiné', 1],
            ['Ananas', 'Saison', 1],
            ['Cerise', 'Rouge', 1],
            ['Cerise', 'Sucré', 1],
            ['Cerise', 'Peut être cuisiné', 1],
            ['Cerise', 'Saison', 1],
            ['Cerise', 'Riche en vitamine C', 1],
            ['Pamplemousse', 'Agrume', 1],
            ['Pamplemousse', 'Acide', 1],
            ['Pamplemousse', 'Saison', 1]
        ],
        'Hiver': [
            ['Kiwi', 'Tropical', 0],
            ['Kiwi', 'Sucré', 1],
            ['Kiwi', 'Acide', 1],
            ['Kiwi', 'Peut être cuisiné', 1],
            ['Kiwi', 'Saison', 0],
            ['Kiwi', 'Riche en vitamine C', 1],
            ['Kiwi', 'Utilisé dans les desserts', 0],
            ['Ananas', 'Tropical', 0],
            ['Ananas', 'Sucré', 1],
            ['Ananas', 'Acide', 1],
            ['Ananas', 'Peut être cuisiné', 1],
            ['Ananas', 'Saison', 0],
            ['Cerise', 'Rouge', 1],
            ['Cerise', 'Sucré', 1],
            ['Cerise', 'Peut être cuisiné', 0],
            ['Cerise', 'Saison', 0],
            ['Cerise', 'Riche en vitamine C', 1],
            ['Pamplemousse', 'Agrume', 1],
            ['Pamplemousse', 'Acide', 1],
            ['Pamplemousse', 'Saison', 0]
        ]
    }
}

def Intent(objects, context):
    attributes = set(context['attributes'])
    for obj in objects:
        obj_rels = [rel[1] for rel in context['relations'][context['seasons'][0]] if rel[0] == obj and rel[2] == 1]
        attributes.intersection_update(set(obj_rels))
    return attributes

def Extent(attributes, context):
    objects = set(context['objects'])
    for att in attributes:
        att_rels = [rel[0] for rel in context['relations'][context['seasons'][0]] if rel[1] == att and rel[2] == 1]
        objects.intersection_update(set(att_rels))
    return objects

def oplus(A, i):
    return sorted(set(A).union(set([i])))

def Next(A, context):
    nbAtt = len(context['attributes'])
    for i in reversed(range(nbAtt)):
        if i not in A:
            B = oplus(A, i)
            if all(j >= i for j in B):
                return B
    return None

def NextClosure(context):
    concepts = []
    A = Extent(Intent(set(), context), context)
    while len(A) < len(context['objects']):
        concepts.append([A, Intent(A, context)])
        A = Next(A, context)
    concepts.append([A, Intent(A, context)])
    return concepts

def concepts_to_array(concepts, context):
    num_objects = len(context['objects'])
    num_attributes = len(context['attributes'])
    result = np.zeros((num_objects, num_attributes))
    
    for i, concept in enumerate(concepts):
        for j, att in enumerate(context['attributes']):
            if att in concept[1]:
                result[i, j] = 1
    
    return result

def build_concept_lattice(concepts):
    lattice = nx.DiGraph()
    for i, concept1 in enumerate(concepts):
        for j, concept2 in enumerate(concepts):
            if i != j and set(concept1[0]).issubset(set(concept2[0])):
                lattice.add_edge(tuple(concept1[0]), tuple(concept2[0]))
    return lattice

def generate_implication_base(concepts, context):
    implications = []
    for concept in concepts:
        premise = set(concept[0])
        conclusion = set(concept[1]) - premise
        if conclusion:
            implications.append((premise, conclusion))
    return implications

def visualize_concept_lattice_3d(lattice, context):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    pos = nx.spring_layout(lattice, dim=3)
    
    for node, (x, y, z) in pos.items():
        ax.scatter(x, y, z, c='r', s=100)
        ax.text(x, y, z, str(node), fontsize=8)
    
    for edge in lattice.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, c='b')
    
    ax.set_title("Treillis de concepts 3D")
    plt.show()

def visualize_concept_heatmap(concepts_array, context):
    plt.figure(figsize=(12, 8))
    plt.imshow(concepts_array, cmap="YlGnBu", aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(context['attributes'])), context['attributes'], rotation=90)
    plt.yticks(range(len(context['objects'])), context['objects'])
    plt.title("Heatmap des concepts")
    plt.tight_layout()
    plt.show()

def visualize_seasonal_comparison(context):
    seasons = context['seasons']
    objects = context['objects']
    attributes = context['attributes']
    
    data = np.zeros((len(objects), len(attributes), len(seasons)))
    
    for i, season in enumerate(seasons):
        for relation in context['relations'][season]:
            obj_index = objects.index(relation[0])
            attr_index = attributes.index(relation[1])
            data[obj_index, attr_index, i] = relation[2]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for i, season in enumerate(seasons):
        im = axes[i].imshow(data[:,:,i], cmap="YlGnBu", aspect='auto')
        axes[i].set_xticks(range(len(attributes)))
        axes[i].set_yticks(range(len(objects)))
        axes[i].set_xticklabels(attributes, rotation=90)
        axes[i].set_yticklabels(objects)
        axes[i].set_title(f"Relations en {season}")
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()

def visualize_implications_network(implications, context):
    G = nx.DiGraph()
    for premise, conclusion in implications:
        for p in premise:
            for c in conclusion:
                G.add_edge(p, c)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, arrows=True)
    nx.draw_networkx_labels(G, pos)
    plt.title("Réseau d'implications")
    plt.axis('off')
    plt.show()

def visualize_attribute_correlation(context):
    correlation_matrix = np.zeros((len(context['attributes']), len(context['attributes'])))
    
    for i, attr1 in enumerate(context['attributes']):
        for j, attr2 in enumerate(context['attributes']):
            count = sum(1 for relation in context['relations']['Été'] + context['relations']['Hiver']
                        if relation[1] in [attr1, attr2] and relation[2] == 1)
            correlation_matrix[i, j] = count
    
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap="YlGnBu")
    plt.colorbar()
    plt.xticks(range(len(context['attributes'])), context['attributes'], rotation=90)
    plt.yticks(range(len(context['attributes'])), context['attributes'])
    plt.title("Corrélation entre les attributs")
    plt.tight_layout()
    plt.show()

def interpret_results(concepts, implications, context):
    print("Analyse des résultats :")
    print(f"Nombre total de concepts : {len(concepts)}")
    print(f"Nombre total d'implications : {len(implications)}")
    
    most_general = [c for c in concepts if len(c[0]) == len(context['objects'])]
    most_specific = [c for c in concepts if len(c[1]) == len(context['attributes'])]
    
    print(f"Concepts les plus généraux : {most_general}")
    print(f"Concepts les plus spécifiques : {most_specific}")
    
    strong_implications = [imp for imp in implications if len(imp[1]) > 1]
    print(f"Implications fortes : {strong_implications}")
    
    # Analyse des fruits par saison
    summer_fruits = set([rel[0] for rel in context['relations']['Été'] if rel[1] == 'Saison' and rel[2] == 1])
    winter_fruits = set([rel[0] for rel in context['relations']['Hiver'] if rel[1] == 'Saison' and rel[2] == 1])
    print(f"Fruits d'été : {summer_fruits}")
    print(f"Fruits d'hiver : {winter_fruits}")
    print(f"Fruits disponibles toute l'année : {summer_fruits.intersection(winter_fruits)}")

# Exécution du code pour générer les concepts et les visualiser
concepts = NextClosure(context_3D)
concepts_array = concepts_to_array(concepts, context_3D)
lattice = build_concept_lattice(concepts)
implications = generate_implication_base(concepts, context_3D)

# Visualisations
visualize_concept_lattice_3d(lattice, context_3D)
visualize_concept_heatmap(concepts_array, context_3D)
visualize_seasonal_comparison(context_3D)
visualize_implications_network(implications, context_3D)
visualize_attribute_correlation(context_3D)

interpret_results(concepts, implications, context_3D)
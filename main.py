import time

def initialize_tamagotchis():
    # Créer une liste de 5 dictionnaires pour les tamagotchis
    return [{'faim': 200, 'santé': 200, 'ennui': 200} for _ in range(5)]

def display_tamagotchis(tamagotchis):
    for i, tam in enumerate(tamagotchis, 1):
        print(f"Tamagotchi {i}: Faim={tam['faim']}, Santé={tam['santé']}, Ennui={tam['ennui']}")

def update_tamagotchis(tamagotchis, elapsed_time):
    # Mettre à jour les niveaux de faim et d'ennui
    for tam in tamagotchis:
        tam['faim'] -= 5 * elapsed_time
        tam['ennui'] -= 3 * elapsed_time
        if tam['ennui'] <= 0:
            # Si l'ennui atteint 0, tous les tamagotchis perdent de la santé
            for t in tamagotchis:
                t['santé'] -= 5 * elapsed_time
        # Vérifier si la partie est perdue
        if tam['faim'] < 0 or tam['santé'] < 0:
            return True
    return False

def perform_action(tamagotchis, action, index):
    if action == 'manger':
        tamagotchis[index]['faim'] += 50
    elif action == 'jouer':
        tamagotchis[index]['ennui'] += 50

def main():
    tamagotchis = initialize_tamagotchis()
    start_time = time.time()
    croquettes = 50

    while True:
        display_tamagotchis(tamagotchis)
        action = input("Action (manger X/jouer X) où X est le numéro du tamagotchi: ")
        action_type, index = action.split()
        index = int(index) - 1

        # Exécuter l'action
        if action_type == 'manger' and croquettes > 0:
            perform_action(tamagotchis, action_type, index)
            croquettes -= 1
        elif action_type == 'jouer':
            perform_action(tamagotchis, action_type, index)

        # Calculer le temps écoulé
        current_time = time.time()
        elapsed_time = int(current_time - start_time)
        start_time = current_time

        # Mettre à jour l'état des tamagotchis et vérifier la fin du jeu
        if update_tamagotchis(tamagotchis, elapsed_time):
            print("La partie est perdue. Un des tamagotchis est mort.")
            break

        # Toutes les 180 secondes (3 minutes)
        if elapsed_time >= 180:
            for tam in tamagotchis:
                tam['santé'] += 50
                tam['ennui'] += 50
            croquettes = 50

if __name__ == '__main__':
    main()
import time
import json  # Import if you choose to use JSON for saving/loading state

def initialize_tamagotchis():
    return [{'faim': 200, 'santé': 200, 'ennui': 200} for _ in range(5)]

def display_tamagotchis(tamagotchis):
    for i, tam in enumerate(tamagotchis, 1):
        print(f"Tamagotchi {i}: Faim={tam['faim']}, Santé={tam['santé']}, Ennui={tam['ennui']}")

def update_tamagotchis(tamagotchis, elapsed_time):
    for tam in tamagotchis:
        tam['faim'] -= 5 * elapsed_time
        tam['ennui'] -= 3 * elapsed_time
        if tam['ennui'] <= 0:
            for t in tamagotchis:
                t['santé'] -= 5 * elapsed_time
        if tam['faim'] < 0 or tam['santé'] < 0:
            return True
    return False

def perform_action(tamagotchis, action, index):
    if action == 'manger':
        tamagotchis[index]['faim'] += 50
    elif action == 'jouer':
        tamagotchis[index]['ennui'] += 50

def save_state(tamagotchis):
    with open('tamagotchi_state.json', 'w') as f:
        json.dump(tamagotchis, f, indent=4)

def load_state():
    try:
        with open('tamagotchi_state.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return initialize_tamagotchis()

def main():
    tamagotchis = load_state()  # Load or initialize new tamagotchis
    start_time = time.time()
    croquettes = 50

    while True:
        display_tamagotchis(tamagotchis)
        try:
            action = input("Action (manger X/jouer X) où X est le numéro du tamagotchi: ")
            action_type, index = action.split()
            index = int(index) - 1
            if action_type in ['manger', 'jouer']:
                perform_action(tamagotchis, action_type, index)
                croquettes -= 1 if action_type == 'manger' and croquettes > 0 else 0
        except ValueError:
            print("Invalid input. Please try again.")
            continue

        current_time = time.time()
        elapsed_time = int(current_time - start_time)
        start_time = current_time

        if update_tamagotchis(tamagotchis, elapsed_time):
            print("La partie est perdue. Un des tamagotchis est mort.")
            break

        if elapsed_time >= 180:
            for tam in tamagotchis:
                tam['santé'] += 50
                tam['ennui'] += 50
            croquettes = 50

        save_state(tamagotchis)  # Save the state after each cycle

if __name__ == '__main__':
    main()
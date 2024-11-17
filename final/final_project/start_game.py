import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.winrate_player import setup_ai as winrate_ai
from agents.RL_player import setup_ai as RL_ai
from agents.console_player import setup_ai as console_ai
import multiprocessing
import os

from baseline.baseline0 import setup_ai as baseline0_ai
from baseline.baseline1 import setup_ai as baseline1_ai
from baseline.baseline2 import setup_ai as baseline2_ai
from baseline.baseline3 import setup_ai as baseline3_ai
from baseline.baseline4 import setup_ai as baseline4_ai  # bluff
from baseline.baseline5 import setup_ai as baseline5_ai
from baseline.baseline6 import setup_ai as baseline6_ai
from baseline.baseline7 import setup_ai as baseline7_ai  # smart

def compete(num):
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    if num == 0:
        config.register_player(name="p1", algorithm=baseline0_ai())
    elif num == 1:
        config.register_player(name="p1", algorithm=baseline1_ai())
    elif num == 2:
        config.register_player(name="p1", algorithm=baseline2_ai())
    elif num == 3:
        config.register_player(name="p1", algorithm=baseline3_ai())
    elif num == 4:
        config.register_player(name="p1", algorithm=baseline4_ai())
    elif num == 5:
        config.register_player(name="p1", algorithm=baseline5_ai())
    elif num == 6:
        config.register_player(name="p1", algorithm=baseline6_ai())
    else:
        config.register_player(name="p1", algorithm=baseline7_ai())

    config.register_player(name="p2", algorithm=winrate_ai())
    game_result = start_poker(config, verbose=1)
    print(json.dumps(game_result, indent=4))
    return game_result["players"][1]["stack"] > game_result["players"][0]["stack"]

if __name__ == '__main__':
    rounds = 300
    # Number of worker processes
    num_workers = 300
    
    # Input data
    data = [5 for _ in range(rounds)]
    
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Map input data to the worker function and collect results
        results = pool.map(compete, data)
    
    win = sum(results)
    lose = rounds - win
    print(f"Win: {win}, Lose: {lose}")


'''
game_log = []

i = 6
for j in range(1):
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
    if i == 0:
        config.register_player(name="p1", algorithm=baseline0_ai())
    elif i == 1:
        config.register_player(name="p1", algorithm=baseline1_ai())
    elif i == 2:
        config.register_player(name="p1", algorithm=baseline2_ai())
    elif i == 3:
        config.register_player(name="p1", algorithm=baseline3_ai())
    elif i == 4:
        config.register_player(name="p1", algorithm=baseline4_ai())
    elif i == 5:
        config.register_player(name="p1", algorithm=baseline5_ai())
    elif i == 6:
        config.register_player(name="p1", algorithm=baseline6_ai())
    else:
        config.register_player(name="p1", algorithm=baseline7_ai())

    config.register_player(name="p2", algorithm=winrate_ai())

    ## Play in interactive mode if uncomment
    #config.register_player(name="me", algorithm=console_ai())
    win, lose = 0, 0
    for k in range(100):
        game_result = start_poker(config, verbose=1)
        print(json.dumps(game_result, indent=4))
        if game_result["players"][1]["stack"] > game_result["players"][0]["stack"]:
            win += 1
        else:
            lose += 1
        print(f"Win: {win}, Lose: {lose}")
    game_log.append({
        "name": f"baseline{i}",
        "win": win,
        "lose": lose
    })

with open("result/winrate_6.json", "w") as file:
    json.dump(game_log, file, indent=4)
'''

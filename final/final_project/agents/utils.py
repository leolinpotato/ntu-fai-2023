import random
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator
import time

def cal_win_rate(hole_card, community_card, players=2, simulation=10000):
    result = [0, 0, 0]  # [win, tie, lose]
    for i in range(simulation):
        result[montecarlo(hole_card, community_card, players)] += 1
    return result[0] / (result[0] + result[2])

def montecarlo(hole_card, community_card, players=2):
    # fill community_card to 5
    community_card = fill_community_card(hole_card, community_card)

    # draw cards for opponents
    used_card_id = [card.to_id for card in (hole_card + community_card)]
    unused_card = [Card.from_id(card) for card in range(1, 53) if card not in used_card_id]
    random.shuffle(unused_card)
    opponents_card = []
    for i in range(players - 1):
        opponents_card.append(unused_card[2*i:2*i+2])

    # compete
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    opponents_score = []
    for i in range(players - 1):
        opponents_score.append(HandEvaluator.eval_hand(opponents_card[i], community_card))
    if my_score > max(opponents_score):
        return 0  # win
    if my_score == max(opponents_score):
        return 1  # tie
    return 2  # lose

def fill_community_card(hole_card, community_card):
    used_card = [card.to_id() for card in (hole_card + community_card)]
    unused_card = [card for card in range(1, 53) if card not in used_card]
    append_id = random.sample(unused_card, 5 - len(community_card))
    return community_card + [Card.from_id(card_id) for card_id in append_id]
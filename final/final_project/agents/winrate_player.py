from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
from agents.utils import *
import ipdb

ALL_IN = 10000

class WinratePlayer(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]   
        if self.net_rating > self.penalty:
            return "fold", 0
        # calculate the money I've spent in this street
        spent = 0
        prev_action = ""
        for action in reversed(round_state["action_histories"][round_state["street"]]):
            if action["uuid"] == self.uuid:
                spent = action["amount"]
                break
            else:
                prev_action = action

        # calculate Break Even rate
        risk = valid_actions[1]["amount"]-spent
        BE_rate = risk/(risk+round_state["pot"]["main"]["amount"])
        print(f"Win: {self.win_rate}, BE: {BE_rate}")

        to_win = self.penalty-self.net_rating  # need how much money to win
        to_earn = (round_state["pot"]["main"]["amount"]+risk)/2  # I can win how much if I win currently
        to_fold = self.net_rating-(to_earn-risk)+(5 if self.SB else 10)  # my net_rating if I fold
    
        # take action based on win_rate
        bet = 0  # bet > 0 -> raise, bet = 0 -> call, bet < 0 -> fold
        r = max(self.net_rating/800, 0)+(-1*round_state["round_count"]**2/2000)+min(self.net_rating/1000, 0)
        print(r)
        # design different strategy for SB and BB
        if self.win_rate >= BE_rate:
            if self.SB:  # more aggresive
                if round_state["street"] == "preflop":
                    if prev_action["action"] == "RAISE":
                        if self.win_rate > 0.7+r:
                            bet = to_win+5
                        elif prev_action["amount"] > self.net_rating+self.penalty and self.net_rating > 0:
                            bet = 0 if self.win_rate > 0.55+r else -1
                        elif prev_action["amount"] > self.net_rating and self.net_rating > 0:
                            bet = 0 if self.win_rate > 0.45+r else -1
                        else:
                            bet = 0 if self.win_rate > 0.35+r else -1
                    else:
                        if self.win_rate > 0.45+r+self.b:
                            bet = self.win_rate*max(100, to_win)
                        elif self.win_rate > 0.35+r+self.b:
                            bet = 30
                else:
                    if prev_action and prev_action["action"] == "RAISE":
                        if self.win_rate > 0.75+1.5*r:
                            bet = ALL_IN
                        elif prev_action["amount"] > self.net_rating and self.net_rating > 0:
                            bet = 0 if self.win_rate > 0.55+r else -1
                        else:
                            bet = 0 if self.win_rate > 0.5+r else -1
                    else:
                        if self.win_rate > 0.65+1.5*r:
                            bet = ALL_IN
                        elif self.win_rate > 0.55+r+self.b/2:
                            bet = to_win*(self.win_rate**2)*2
                        elif self.win_rate > 0.45+r+self.b/2:
                            bet = to_win*(self.win_rate**2)
            else:
                if round_state["street"] == "preflop":
                    if prev_action["action"] == "RAISE":
                        if self.win_rate > 0.7+r:
                            bet = to_win+5
                        elif prev_action["amount"] > self.net_rating+self.penalty and self.net_rating > 0:
                            bet = 0 if self.win_rate > 0.6+r else -1
                        elif prev_action["amount"] > self.net_rating and self.net_rating > 0:
                            bet = 0 if self.win_rate > 0.45+r else -1
                        else:
                            bet = 0 if self.win_rate > 0.35+r else -1
                    else:
                        if self.win_rate > 0.5+r:
                            bet = to_win+5
                        elif self.win_rate > 0.4+r:
                            bet = self.win_rate*max(100, to_win)
                        elif self.win_rate > 0.3+r:
                            bet = 30
                else:
                    if prev_action["action"] == "RAISE":
                        if self.win_rate > 0.7+1.5*r:
                            bet = ALL_IN
                        elif prev_action["amount"] > self.net_rating+self.penalty and self.net_rating > 0:
                            bet = 0 if self.win_rate > 0.6+r else -1
                        elif prev_action["amount"] > self.net_rating and self.net_rating > 0:
                            bet = 0 if self.win_rate > 0.5+r else -1
                        else:
                            bet = 0 if self.win_rate > 0.45+r else -1
                    else:
                        if self.win_rate > 0.6+1.5*r:
                            bet = ALL_IN
                        elif self.win_rate > 0.45+r:
                            bet = to_win*(self.win_rate**2)
                        elif self.win_rate > 0.4+r:
                            bet = to_win*(self.win_rate**2)*0.5
        else:
            bet = -1

        if bet > 0:
            if to_earn > to_win:
                action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
            elif valid_actions[2] and valid_actions[2]["amount"]["max"] != -1:
                action, amount = valid_actions[2]["action"], min(max(bet, valid_actions[2]["amount"]["min"]), valid_actions[2]["amount"]["max"])
            else:
                action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        elif to_fold < -1*(self.penalty-5):  # if I lose this round I'll lose -> all in
            if valid_actions[2] and valid_actions[2]["amount"]["max"] != -1:
                action, amount = valid_actions[2]["action"], valid_actions[2]["amount"]["max"]
            else:
                action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        elif bet == 0:
            action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        elif bet == -1:
            action, amount = valid_actions[0]["action"], valid_actions[0]["amount"]

        return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        self.players = game_info["player_num"]
        self.bluff_cnt = 0
        self.action_cnt = 0
        # find out my position in this game
        for i, seat in enumerate(game_info["seats"]):
            if seat["uuid"] == self.uuid:
                self.pos = i
                break

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole = [Card.from_str(card) for card in hole_card]
        # calculate the money needed to win even if I fold all following rounds
        for i, seat in enumerate(seats):
            if seat["uuid"] == self.uuid:
                self.net_rating = seat["stack"] - 1000
                break
        self.b = 0.2 if self.action_cnt > 0 and self.bluff_cnt/self.action_cnt > 1/4 else 0

    def receive_street_start_message(self, street, round_state):
        # find out I'm SB or BB in this round
        if round_state["small_blind_pos"] == self.pos:
            self.SB = True
        else:
            self.SB = False

        self.penalty = ((20-round_state["round_count"])//2)*15
        if round_state["round_count"] % 2:
            if self.SB:
                self.penalty += 10
            else:
                self.penalty += 5
        if self.net_rating > self.penalty:
            return
        # calculate the win rate depending on hole cards and community cards
        community_card = round_state["community_card"]
        self.community = [Card.from_str(card) for card in community_card]
        self.win_rate = cal_win_rate(self.hole, self.community, self.players, 10000)


    def receive_game_update_message(self, action, round_state):
        # to handle the player who tends to bluff
        if action["player_uuid"] != self.uuid:
            self.action_cnt += 1
            if action["action"] == "raise":
                self.bluff_cnt += 1

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return WinratePlayer()

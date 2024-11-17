# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
 
        "*** YOUR CODE HERE ***"
        original_score = successorGameState.getScore()

        additional_score = 0
        # should keep away from ghost
        for ghost in newGhostStates:
            distance = manhattanDistance(newPos, ghost.getPosition())
            if distance <= 1:
                additional_score -= 500
        # should go for food
        food_list = newFood.asList()
        min_distance = 1e5 if food_list else 0
        for food in food_list:
            min_distance = min(manhattanDistance(newPos, food), min_distance)
        additional_score -= min_distance * 0.1

        return original_score + additional_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        scores = [self.value(gameState.generateSuccessor(0, legalMove), 1, 0) for legalMove in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def value(self, gameState: GameState, agent, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.max_value(gameState, agent, depth)
        else:
            return self.min_value(gameState, agent, depth)

    def max_value(self, gameState: GameState, agent, depth):
        v = float("-inf")
        legalMoves = gameState.getLegalActions(agent)
        nextGameStates = [gameState.generateSuccessor(agent, legalMove) for legalMove in legalMoves]
        for nextGameState in nextGameStates:
            v = max(v, self.value(nextGameState, agent + 1, depth))
        return v

    def min_value(self, gameState: GameState, agent, depth):
        v = float("inf")
        legalMoves = gameState.getLegalActions(agent)
        nextGameStates = [gameState.generateSuccessor(agent, legalMove) for legalMove in legalMoves]
        if agent == gameState.getNumAgents() - 1:
            agent = -1
            depth += 1
        for nextGameState in nextGameStates:
            v = min(v, self.value(nextGameState, agent + 1, depth))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        a, b = float("-inf"), float("inf")
        scores = []
        bestScore = float("-inf")
        for legalMove in legalMoves:
            score = self.value(gameState.generateSuccessor(0, legalMove), 1, 0, a, b)
            scores.append(score)
            if score > a:
                a = score
        bestIndices = [index for index in range(len(scores)) if scores[index] == a]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def value(self, gameState: GameState, agent, depth, a, b):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.max_value(gameState, agent, depth, a, b)
        else:
            return self.min_value(gameState, agent, depth, a, b)

    def max_value(self, gameState: GameState, agent, depth, a, b):
        v = float("-inf")
        legalMoves = gameState.getLegalActions(agent)
        for legalMove in legalMoves:
            nextGameState = gameState.generateSuccessor(agent, legalMove)
            v = max(v, self.value(nextGameState, agent + 1, depth, a, b))
            if v > b:
                return v
            a = max(a, v)
        return v

    def min_value(self, gameState: GameState, agent, depth, a, b):
        v = float("inf")
        legalMoves = gameState.getLegalActions(agent)
        if agent == gameState.getNumAgents() - 1:
            agent = -1
            depth += 1
        for legalMove in legalMoves:
            nextGameState = gameState.generateSuccessor(agent, legalMove)
            v = min(v, self.value(nextGameState, agent + 1, depth, a, b))
            if v < a:
                return v
            b = min(b, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        scores = [self.value(gameState.generateSuccessor(0, legalMove), 1, 0) for legalMove in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def value(self, gameState: GameState, agent, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.max_value(gameState, agent, depth)
        else:
            return self.exp_value(gameState, agent, depth)

    def max_value(self, gameState: GameState, agent, depth):
        v = float("-inf")
        legalMoves = gameState.getLegalActions(agent)
        nextGameStates = [gameState.generateSuccessor(agent, legalMove) for legalMove in legalMoves]
        for nextGameState in nextGameStates:
            v = max(v, self.value(nextGameState, agent + 1, depth))
        return v

    def exp_value(self, gameState: GameState, agent, depth):
        v = 0
        legalMoves = gameState.getLegalActions(agent)
        nextGameStates = [gameState.generateSuccessor(agent, legalMove) for legalMove in legalMoves]
        if agent == gameState.getNumAgents() - 1:
            agent = -1
            depth += 1
        for nextGameState in nextGameStates:
            v += self.value(nextGameState, agent + 1, depth)
        return v / len(nextGameStates)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()

    original_score = scoreEvaluationFunction(currentGameState)

    additional_score = 0
    # should keep away from ghost
    for i, ghost in enumerate(newGhostStates):
        distance = manhattanDistance(newPos, ghost.getPosition())
        cost = 500 if distance <= 5 else 0
        if newScaredTimes[i] < 5:
            additional_score -= cost
    # should go for food
    food_list = newFood.asList()
    min_food_distance = 1e5 if food_list else 0
    for food in food_list:
        min_food_distance = min(manhattanDistance(newPos, food), min_food_distance)
    additional_score -= min_food_distance
    # try to eat capsules
    additional_score -= len(newCapsules) * 100

    return original_score + additional_score

# Abbreviation
better = betterEvaluationFunction

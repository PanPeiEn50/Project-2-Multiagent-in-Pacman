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
        # Evaluate the food
        foodPos = newFood.asList()
        foodCount = len(foodPos)
        closestFoodDistance = float('inf')
        for pos in foodPos:
            distance = manhattanDistance(pos, newPos)
            if distance < closestFoodDistance:
                closestFoodDistance = distance

        # avoid divide zero
        if foodCount == 0:
            closestFoodDistance = 0

        # Evaluate the ghost
        closestGhostDistance = float('inf')
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            if distance < closestGhostDistance:
                closestGhostDistance = distance

        # Evaluate the scared ghost
        minScaredTime = min(newScaredTimes) if newScaredTimes else 0

        # Calculate the score
        score = successorGameState.getScore()

        # Reward closer food
        score += 2.0 / (closestFoodDistance + 1) - 1.5 * foodCount

        # High penalty if close to the ghost
        if closestGhostDistance <= 1:
            score -= 200
        else:
            score -= 10 / (closestGhostDistance + 1)

        # Reward if the scared ghost nearby
        if minScaredTime > 0:
            score += 200 / (closestGhostDistance + 1)

        return score

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
        def minimax(state, depth, agentIndex):
        # Base case
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Number of agents
            numAgents = state.getNumAgents()

            # Next agent index and depth
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions = state.getLegalActions(agentIndex)

            # Remove 'Stop' action to prevent Pacman from stopping
            if 'Stop' in actions:
                actions.remove('Stop')

            # Recursive Minimax calls
            if agentIndex == 0:  # Maximizer
                return max(minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in actions)
            else:  # Minimizer
                return min(minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in actions)

        # Initiate Minimax from Pacman's perspective and find the best action
        pacmanActions = gameState.getLegalActions(0)
        if 'Stop' in pacmanActions:
            pacmanActions.remove('Stop')

        scores = [minimax(gameState.generateSuccessor(0, action), 0, 1) for action in pacmanActions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0]

        return pacmanActions[chosenIndex]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Base case
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:  # Maximizer
                value = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    if action != 'Stop':
                        successor = state.generateSuccessor(agentIndex, action)
                        value = max(value, alphaBeta(successor, nextDepth, nextAgent, alpha, beta))
                        if value > beta:
                            return value
                        alpha = max(alpha, value)
                return value
            else:  # Minimizer
                value = float("inf")
                for action in state.getLegalActions(agentIndex):
                    if action != 'Stop':
                        successor = state.generateSuccessor(agentIndex, action)
                        value = min(value, alphaBeta(successor, nextDepth, nextAgent, alpha, beta))
                        if value < alpha:
                            return value
                        beta = min(beta, value)
                return value

        bestScore = float("-inf")
        bestAction = None
        alpha = float("-inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):
            if action != 'Stop':
                successor = gameState.generateSuccessor(0, action)
                score = alphaBeta(successor, 0, 1, alpha, beta)  # Start with depth 0 and agent 1
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                alpha = max(alpha, bestScore)

        return bestAction

        util.raiseNotDefined()

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
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            # Base case
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            # remove "Stop"
            legalActions = [action for action in state.getLegalActions(agentIndex) if action != 'Stop']

            if agentIndex != 0:
                scores = [expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in legalActions]
                return sum(scores) / len(scores)  # Return average score

            else:
                scores = [expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in legalActions]
                if depth == 0:
                    return legalActions[scores.index(max(scores))]
                return max(scores)

        return expectimax(gameState, 0, 0)

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()

    # Calculate the impact of nearby ghost
    def ghostImpact(gameState):
        penalty = 0
        for ghost in gameState.getGhostStates():
            distance = manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition())
            if ghost.scaredTimer > 0:
                # Encourage Pacman to chase scared ghosts
                penalty -= 200 / max(distance, 1)
            else:
                # Penalize when too close to ghost
                penalty += 2 * max(10 - distance, 0)
        return penalty

    # Calculate the benefit of remaining food
    def foodBenefit(gameState):
        foodDistances = [manhattanDistance(gameState.getPacmanPosition(), food) for food in gameState.getFood().asList()]
        if foodDistances:
            return -min(foodDistances)  
        return 0

    # Calculate the benefit of capsules
    def capsuleBenefit(gameState):
        capsuleDistances = [manhattanDistance(gameState.getPacmanPosition(), cap) for cap in gameState.getCapsules()]
        if capsuleDistances:
            return -min(capsuleDistances) * 2  
        return 0

    score += ghostImpact(currentGameState)
    score += foodBenefit(currentGameState)
    score += capsuleBenefit(currentGameState)

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

#!/usr/bin/env python2

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

from collections import defaultdict

UPDATE_STRING = 'LearningAgent.update(): deadline = {}, inputs = {}, ' + \
                'action = {}, reward = {}'

LEARNING_RATE = 0.1
DISCOUNT = 0.5
EPSILON = 0.1

class HardCodedAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None
        super(HardCodedAgent, self).__init__(env)
        self.color = 'white'  # override color
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here
        self.next_waypoint = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.next_waypoint = random.choice(Environment.valid_actions)

    def update(self, t):
        # Gather inputs from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        print(inputs)

        action_okay = True
        light = inputs['light']
        if self.next_waypoint == 'right':
            if light == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'straight':
            if light == 'red':
                action_okay = False
        elif self.next_waypoint == 'left':
            if light == 'red' or \
               (
                   inputs['oncoming'] == 'forward' or
                   inputs['oncoming'] == 'right'
               ):
                action_okay = False

        action = None
        if action_okay:
            action = self.next_waypoint

        # TODO: Update state

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print(UPDATE_STRING).format(deadline, inputs, action, reward)


class RandomAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None
        super(LearningAgent, self).__init__(env)
        self.color = 'white'  # override color
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here
        self.next_waypoint = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # do nothing for now

        # For now we only generate random actions
        action = random.choice(Environment.valid_actions[1:])

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print(UPDATE_STRING).format(deadline, inputs, action, reward)


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None
        super(LearningAgent, self).__init__(env)
        self.color = 'white'  # override color
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.q = defaultdict(int)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.destination = destination
        self.last_reward = None
        self.last_action = None
        self.last_state = None
        self.state = None

    @staticmethod
    def _get_action_index(action):
        return Environment.valid_actions.index(action)

    def best_action(self, state):
        # Find all Q values that share the same state we have here
        candidates = [k for k in self.q.keys() if k[0] == state]
        # if there are no candidates, we have to return a random action
        if len(candidates) == 0:
            return 0, random.choice(Environment.valid_actions)
        # Build a list we can sort to choose the action with highest value
        qs = [(self.q[c], c) for c in candidates]
        qs = sorted(qs, key=lambda x: x[0], reverse=True)
        # Now we return the first action as we sorted by descending value
        # The list's first position (0) is a pair (value, state-action),
        # so we extract the action
        return qs[0][0], qs[0][1][-1]

    def get_action(self, state, best_action):
        if random.random() < EPSILON:
            return random.choice(Environment.valid_actions)
        else:
            return best_action[1]

    def update(self, t):
        # Gather inputs
        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        current_state = (
            inputs['light'] == 'red',
            self._get_action_index(inputs['oncoming']),
            self._get_action_index(inputs['left']),
            self._get_action_index(inputs['right']),
            self._get_action_index(self.next_waypoint),
        )

        best_action = self.best_action(current_state)
        action = self.get_action(current_state, best_action)

        # Execute action and get reward
        reward = self.env.act(self, action)

        if self.last_state:
            # If this is True we are not in the first iteration
            self.q[self.last_state, self.last_action] += LEARNING_RATE * (
                self.last_reward + DISCOUNT * best_action[0] -
                self.q[self.last_state, self.last_action]
            )

        self.last_state = self.state
        self.state = current_state
        self.last_action = action
        self.last_reward = reward
        print('----')
        print(self.q)
        print(self.state)
        print(best_action)
        print(UPDATE_STRING).format(deadline, inputs, action, reward)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, frame_delay=1)
    sim.run(n_trials=5)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()

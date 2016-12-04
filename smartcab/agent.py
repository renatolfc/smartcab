#!/usr/bin/env python2

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

UPDATE_STRING = 'LearningAgent.update(): deadline = {}, inputs = {}, ' + \
                'action = {}, reward = {}'


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

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.destination = destination
        self.state = [0] * 5
        # TODO: Prepare for a new trip; reset any variables here, if required

    @staticmethod
    def _get_action_index(action):
        return Environment.valid_actions.index(action)

    def update(self, t):
        # Gather inputs
        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # deadline = 18, inputs = {'light': 'red', 'oncoming': None, 'right':
        # None, 'left': None},action = right, reward = 1
        self.state = [
            deadline, deadline < 0, inputs['light'] == 'red',
            self._get_action_index(inputs['oncoming']),
            self._get_action_index(inputs['left']),
            self._get_action_index(inputs['right']),
        ]

        # For now we only generate random actions
        action = random.choice(Environment.valid_actions[1:])

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print(UPDATE_STRING).format(deadline, inputs, action, reward)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e)
    sim.run(n_trials=5)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()

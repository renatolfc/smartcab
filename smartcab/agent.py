import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

OPTIMIZED=True
random.seed(12345)

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        self.trial = -1
        self.last_state = None


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        self.trial += 1

        ###########
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0

        if testing:
            self.epsilon = 0.0
            self.alpha = 0.0
        else:
            if not OPTIMIZED:
                if self.trial == 0:
                    return
                self.epsilon = self.epsilon - 0.05
            else:
                if self.trial == 0:
                    return
                #self.epsilon = self.epsilon - (0.0001 * self.trial)**2
                #self.epsilon = math.exp(-0.025 * self.trial)
                self.epsilon = math.cos(0.0125 * self.trial)
                #self.epsilon = self.epsilon - 0.001
                #if self.epsilon > 0.5:
                #self.epsilon = (200 - self.trial) / 200
                #else:
                #    self.epsilon = self.epsilon / 2

            #self.epsilon = self.epsilon / 2
            #self.epsilon = math.exp(-self.epsilon * self.trial)
            #self.epsilon = math.exp(-0.07 * self.trial)  # promete
            #self.epsilon = self.epsilon - 0.05
            #self.epsilon = 0.90 ** (self.trial)
            #self.epsilon -= 0.01

        return None


    def build_state(self):
        """ The build_state function is called when the agent requests data from the
            environment. The next waypoint, the intersection inputs, and the deadline
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ###########
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        # When learning, check if the state is in the Q-table
        #   If it is not, create a dictionary in the Q-table for the current 'state'
        #   For each action, set the Q-value for the state-action pair to 0

        light = inputs['light']
        can_go_forward = light != 'red'
        can_go_left = light != 'red' and inputs['oncoming'] != 'forward'
        can_go_right = light != 'red' or (inputs['oncoming'] != 'left' and
                                          inputs['left'] != 'forward')

        can_go = False
        if waypoint == 'forward':
            can_go = can_go_forward
        elif waypoint == 'right':
            can_go = can_go_right
        else:
            can_go = can_go_left

        state = (
            light != 'red',
            waypoint,
            can_go_forward and waypoint == 'forward',
            can_go_right and waypoint == 'right',
            can_go_left and waypoint == 'left',
        )

        #state = (
        #    light,
        #    waypoint,
        #    inputs['oncoming'],
        #)


#        state = (
#            light,
#            inputs['oncoming'],
#            inputs['left'],
#            inputs['right'],
#            waypoint,
#            can_go
#        )

        #state = (
        #    light == 'red',
        #    self.valid_actions.index(waypoint),
        #    self.valid_actions.index(inputs['oncoming']),
        #    self.valid_actions.index(inputs['right']),
        #    self.valid_actions.index(inputs['left']),
        #    deadline
        #)


        if self.learning and state not in self.Q:
            self.createQ(state)

        self.last_state = state
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        if state not in self.Q:
            assert not self.learning
            return random.choice(self.valid_actions)


        # Build a list we can sort to choose the action with highest value
        qs = sorted(self.Q[state].items(), key=lambda x: x[-1], reverse=True)
        i = 1
        while qs[i][0] == qs[0][0] and i < len(qs) - 1:
            i += 1
        best = random.choice(qs[0:i])[0]
        # Now we return the first action as we sorted by descending value
        # The list's first position (0) is a pair (value, state-action),
        # so we extract the action
        return best


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ###########
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.learning:
            if state not in self.Q:
                self.Q[state] = {a: 0.0 for a in self.valid_actions}


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state

        if not self.learning:
                return random.choice(self.valid_actions)

        if random.random() < self.epsilon:
            return random.choice(self.valid_actions)
        else:
            return self.get_maxQ(state)

        #if self.learning:
        #    # When learning, choose a random action with 'epsilon' probability
        #    # Otherwise, choose an action with the highest Q-value for the
        #    # current state
        #    if random.random() < self.epsilon:
        #        action = random.choice(self.valid_actions)
        #    else:
        #        action = self.get_maxQ(state)
        #else:
        #    # When not learning, choose a random action
        #    action = random.choice(self.valid_actions)

        #return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards
            when conducting learning. """

        if not self.learning:
            return

        self.Q[state][action] = (1-self.alpha) * self.Q[state][action] + \
                                self.alpha * reward


    def update(self):
        """ The update function is called when a time step is completed in the
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return


def run():
    """ Driving function for running the simulation.
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    if not OPTIMIZED:
        env = Environment(verbose=False)
        agent = env.create_agent(LearningAgent, learning=True, epsilon=1.0, alpha=0.5)
        env.set_primary_agent(agent, enforce_deadline=True)
        sim = Simulator(env, update_delay=0, log_metrics=True, optimized=False, display=False)
        sim.run(tolerance=0.05, n_test=10)
    else:
        ##############
        # Create the environment
        # Flags:
        #   verbose     - set to True to display additional output from the simulation
        #   num_dummies - discrete number of dummy agents in the environment, default is 100
        #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
        env = Environment(verbose=False, grid_size=(8, 6))

        ##############
        # Create the driving agent
        # Flags:
        #   learning   - set to True to force the driving agent to use Q-learning
        #    * epsilon - continuous value for the exploration factor, default is 1
        #    * alpha   - continuous value for the learning rate, default is 0.5
        agent = env.create_agent(LearningAgent, learning=True, epsilon=1.0, alpha=0.4)


        ##############
        # Follow the driving agent
        # Flags:
        #   enforce_deadline - set to True to enforce a deadline metric
        env.set_primary_agent(agent, enforce_deadline=True)

        ##############
        # Create the simulation
        # Flags:
        #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
        #   display      - set to False to disable the GUI if PyGame is enabled
        #   log_metrics  - set to True to log trial and simulation results to /logs
        #   optimized    - set to True to change the default log file name
        sim = Simulator(env, update_delay=0, log_metrics=True, optimized=True, display=False)

        ##############
        # Run the simulator
        # Flags:
        #   tolerance  - epsilon tolerance before beginning testing, default is 0.05
        #   n_test     - discrete number of testing trials to perform, default is 0
        sim.run(tolerance=0.001, n_test=10)


if __name__ == '__main__':
    run()

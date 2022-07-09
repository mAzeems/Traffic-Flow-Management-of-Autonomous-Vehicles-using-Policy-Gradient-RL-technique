import traci
import numpy as np
import random
import math
from pyglet.window import key

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7


class SimRunner:
    def __init__(self, sess, model, memory, traffic_gen, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd):
        self._sess = sess
        self._model = model
        self._memory = memory
        self._traffic_gen = traffic_gen
        self._total_episodes = total_episodes
        self._gamma = gamma
        self._eps = 0  # controls the explorative/exploitative payoff, I choosed epsilon-greedy policy
        self._steps = 0
        self._waiting_times = {}
        self._sumoCmd = sumoCmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._sum_intersection_queue = 0
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_intersection_queue_store = []


    # THE MAIN FUCNTION WHERE THE SIMULATION HAPPENS
    def run(self, episode):
        # first, generate the route file for this simulation and set up sumo
        self._traffic_gen.generate_routefile(episode)
        traci.start(self._sumoCmd)

        # set the epsilon for this episode
        self._eps = 1.0 - (episode / self._total_episodes)
        # inits
        self._steps = 0
        tot_neg_reward = 0
        old_total_wait = 0
        curr_policy=0 
        updated_policy=0
        self._waiting_times = {}
        self._sum_intersection_queue = 0

        while self._steps < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._get_waiting_times()
            reward = old_total_wait - current_total_wait
            # saving the data into the memory


            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            if reward < 0:
                tot_neg_reward += reward

        self._save_stats(tot_neg_reward)
        print("Total reward: {}, Eps: {}".format(tot_neg_reward, self._eps))
        traci.close()

    # HANDLE THE CORRECT NUMBER OF STEPS TO SIMULATE
    def _simulate(self, steps_todo):
        if (self._steps + steps_todo) >= self._max_steps:  # do not do more steps than the maximum number of steps
            steps_todo = self._max_steps - self._steps
        self._steps = self._steps + steps_todo  # update the step counter
        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._replay()  # training
            steps_todo -= 1
            intersection_queue = self._get_stats()
            self._sum_intersection_queue += intersection_queue

    # RETRIEVE THE WAITING TIME OF EVERY CAR IN THE INCOMING LANES
    def _get_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        for veh_id in traci.vehicle.getIDList():
            wait_time_car = traci.vehicle.getAccumulatedWaitingTime(veh_id) #	Returns the accumulated waiting time [s] within the previous time interval of default length 100 s. (length is configurable per option --waiting-time-memory given to the main application)
            road_id = traci.vehicle.getRoadID(veh_id)  # get the road id where the car is located 	Returns the id of the edge the named vehicle was at within the last step; error value: ""
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[veh_id] = wait_time_car
            else:
                if veh_id in self._waiting_times:
                    del self._waiting_times[veh_id]  # the car isnt in incoming roads anymore, delete his waiting time
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    # DECIDE WHETER TO PERFORM AN EXPLORATIVE OR EXPLOITATIVE ACTION = EPSILON-GREEDY POLICY
    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1) # random action #exploration 
        else:
            return np.argmax(self._model.predict_one(state, self._sess)) # the best action given the current state #exploitation

    # SET IN SUMO THE CORRECT YELLOW PHASE
    def _set_yellow_phase(self, old_action):
        yellow_phase = old_action * 2 + 1 # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase("TL", yellow_phase)

    # SET IN SUMO A GREEN PHASE
    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    # RETRIEVE THE STATS OF THE SIMULATION FOR ONE SINGLE STEP
    def _get_stats(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL") #Returns the total number of halting vehicles for the last time step on the given edge. A speed of less than 0.1 m/s is considered a halt.
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        intersection_queue = halt_N + halt_S + halt_E + halt_W
        return intersection_queue

    # RETRIEVE THE STATE OF THE INTERSECTION FROM SUMO
    def _get_state(self):
        state = np.zeros(self._model.num_states)

        for veh_id in traci.vehicle.getIDList(): #Returns a list of ids of all vehicles currently running within the scenario (the given vehicle ID is ignored)
            lane_pos = traci.vehicle.getLanePosition(veh_id) #The position of the vehicle along the lane (the distance from the front bumper to the start of the lane in [m]); error value: -2^30
            lane_id = traci.vehicle.getLaneID(veh_id) #Returns the id of the lane the named vehicle was at within the last step; error value: ""
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to TL, lane_pos = 0
            lane_group = -1  # just dummy initialization
            valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            # distance in meters from the TLS -> mapping into cells
            if lane_pos < 10:
                lane_cell = 0
            elif lane_pos < 15:
                lane_cell = 1
            elif lane_pos < 20:
                lane_cell = 2
            elif lane_pos < 30:
                lane_cell = 3
            elif lane_pos < 45:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 120:
                lane_cell = 6
            elif lane_pos < 180:
                lane_cell = 7
            elif lane_pos < 360:
                lane_cell = 8
            elif lane_pos <= 800:
                lane_cell = 9

            # finding the lane where the car is located - _3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7

            if lane_group >= 1 and lane_group <= 7:
                veh_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                veh_position = lane_cell
                valid_car = True

            if valid_car:
                state[veh_position] = 1  # write the position of the car veh_id in the state array

        return state

    def _replay(self):
        batch = self._memory.get_samples(self._model.batch_size)
        if len(batch) > 0:  # if there is at least 1 sample in the batch
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            p_state = self._model.predict_batch(states, self._sess)  # predict state, for every sample
            p_next_state = self._model.predict_batch(next_states, self._sess)  # predict next_state, for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._model.num_states))
            y = np.zeros((len(batch), self._model.num_actions))

            for i, b in enumerate(batch):
                state, action, reward, next_state = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_state = p_state[i]  # get the state predicted before
                current_state[action] = reward + self._gamma * np.amax(p_next_state[i])  # update state, action
                x[i] = state
                y[i] = current_state  # state that includes the updated policy value

            self._model.train_batch(self._sess, x, y)  # train the NN
        def policy(self):
            batch = self._memory.get_samples(self._model.batch_size)
            curr_policy = self._sess
            updated_policy = np.array([val[3] for val in batch])

    # SAVE THE STATS OF THE EPISODE TO PLOT THE GRAPHS AT THE END OF THE SESSION
    def _save_stats(self, tot_neg_reward):
            self._reward_store.append(tot_neg_reward)  # how much negative reward in this episode
    @property
    def reward_store(self):
        return self._reward_store

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 1.0
            if self.move[1]: u[2] += 1.0
            if self.move[3]: u[3] += 1.0
            if self.move[2]: u[4] += 1.0
            if True not in self.move:
                u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True
    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False

    

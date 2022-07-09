from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from sumolib import checkBinary
import matplotlib.pyplot as plt
import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import timeit
from SimRunner import SimRunner
from Model import TrafficGenerator
from Model import Memory
from Model import Model
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# PLOT AND SAVE THE STATS ABOUT THE SESSION
def save_graphs(sim_runner, total_episodes, plot_path):

    plt.rcParams.update({'font.size': 24})  # set bigger font size

    # reward
    data = sim_runner.reward_store
    plt.plot(data)
    plt.ylabel("Cumulative negative reward")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val + 0.05 * min_val, max_val - 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'reward.png', dpi=96)
    plt.close("all")
    with open(plot_path + 'reward_data.txt', "w") as file:
        for item in data:
                file.write("%s\n" % item)

if __name__ == "__main__":

    # --- TRAINING OPTIONS ---
    gui = True
    total_episodes = 100
    gamma = 0.99
    batch_size = 2400
    memory_size = 45000
    path = "./results/"

    # attributes of the agent
    num_states = 256
    num_actions = 4
    max_steps = 4800
    green_duration = 30
    yellow_duration = 45
    
    # setting the cmd mode or the visual mode
    if gui == True:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # initializations
    model = Model(num_states, num_actions, batch_size)
    memory = Memory(memory_size)
    traffic_gen = TrafficGenerator(max_steps)
    sumoCmd = [sumoBinary, "-c", "intersection/tlcs_config_train.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("PATH:", path)
        print("----- Start time:", datetime.datetime.now())
        sess.run(model.var_init)
        sim_runner = SimRunner(sess, model, memory, traffic_gen, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd)
        episode = 0

        while episode < total_episodes:
            print('----- Episode {} of {}'.format(episode+1, total_episodes))
            start = timeit.default_timer()
            sim_runner.run(episode)  # run the simulation
            stop = timeit.default_timer()
            print('Time: ', round(stop - start, 1))
            episode += 1

        os.makedirs(os.path.dirname(path), exist_ok=True)
        saver.save(sess, path + "my_tlcs_model.ckpt") 
        print("----- End time:", datetime.datetime.now())
        print("PATH:", path)
        save_graphs(sim_runner, total_episodes, path)

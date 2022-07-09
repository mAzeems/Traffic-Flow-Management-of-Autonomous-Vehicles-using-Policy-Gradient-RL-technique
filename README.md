# RL-controlled-Lights-and-I2V-SUMO
This is the portion of our Final Year Project in Bachelors. We try to avoid congestion on two levels i.e Intersection level and Infrastructure to Vehicle communication level. 

We used ANN to approximate Bellman Q-Learning Optimality equation.

For I2V communication we used Lane Area Detectors which is the analogy of Vehicle Tracking Cameras on roads.
We collected the information of all detectors from all regions and conveyed that information to Upcoming traffic so that they can take less congested path.

In tlcs_main.py file set gui = True to start training of NN.

Once training is finished, run the SUMO simulation. it will give you results of RL only.

Then run the python script "lane_area_detectors.py" . It will give you combined results of RL and I2V.

Good Luck! 

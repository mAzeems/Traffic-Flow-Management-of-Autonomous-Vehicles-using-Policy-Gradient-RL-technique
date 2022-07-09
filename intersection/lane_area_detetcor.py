# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 03:08:46 2020

@author: usman
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:43:28 2019

@author: usman
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:42:48 2019

@author: usman
"""

import os, sys
import optparse
 

if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")
     
from sumolib import checkBinary 
import traci 

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
    default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


# contains TraCI control loop
def run():
    counter = 0
    #checker=True
    while traci.simulation.getMinExpectedNumber() > 0:
        
        traci.simulationStep()
        
        
        
        if counter<1000:
            
            veh_id = "veh_" + str(counter)
            #print(veh_id)
            route_id= traci.vehicle.getRouteID(veh_id)
            #print (route_id)
            counter+=1
        
        
        
            if route_id in ["W_N_E" , "N_S_W", "N_W_E", "W_N_S"]:
                #print("hello N2TL")#N2TL
                #print(traci.lanearea.getJamLengthMeters('N_S_1'),traci.lanearea.getJamLengthVehicle('N_S_1'))
                if traci.lanearea.getJamLengthMeters('N_S_1')>=40 or traci.lanearea.getJamLengthMeters('N_S_2')>=40 or traci.lanearea.getJamLengthMeters('N_S_3')>=40 or traci.lanearea.getJamLengthMeters('N_S_4')>=40:
                    traci.vehicle.setColor(veh_id, (1,1,0))
                    
                    if route_id == "N_W_E":
                        traci.vehicle.setRouteID(veh_id,'N_E')
                    if route_id == "N_S_W":
                        traci.vehicle.setRouteID(veh_id,'N_W')
                    if route_id == "W_N_E":
                        traci.vehicle.setRouteID(veh_id,'W_E')
                    if route_id == "W_N_S":
                        traci.vehicle.setRouteID(veh_id,'W_S')
                    
            if route_id in ["W_S_N" , "E_S_N", "S_N_W","E_S_W"]:
                #print("hello S2TL")#S2TL
                #print(traci.lanearea.getJamLengthMeters('S_N_1'),traci.lanearea.getJamLengthVehicle('S_N_1'))
                if traci.lanearea.getJamLengthMeters('S_N_1')>=40 or traci.lanearea.getJamLengthMeters('S_N_2')>=40 or traci.lanearea.getJamLengthMeters('S_N_3')>=40 or traci.lanearea.getJamLengthMeters('S_N_4')>=40:
                    traci.vehicle.setColor(veh_id, (1,1,0))
                    #print(counter)
                    if route_id == "W_S_N":
                        traci.vehicle.setRouteID(veh_id,'W_N')
                    if route_id == "E_S_N":
                        traci.vehicle.setRouteID(veh_id,'E_N')
                    if route_id == "S_N_W":
                        traci.vehicle.setRouteID(veh_id,'S_W')
                    if route_id == "E_S_W":
                        traci.vehicle.setRouteID(veh_id,'E_W')

            if route_id in ["S_E_N" , "S_E_W", "E_W_N","E_N_S", "N_E_S"]:
                #print("hello E2TL")#E2TL
                #print(traci.lanearea.getJamLengthMeters('E_W_1'),traci.lanearea.getJamLengthVehicle('E_W_1'))
                if traci.lanearea.getJamLengthMeters('E_W_1')>=40 or traci.lanearea.getJamLengthMeters('E_W_2')>=40 or traci.lanearea.getJamLengthMeters('E_W_3')>=40 or traci.lanearea.getJamLengthMeters('E_W_4')>=40:
                    traci.vehicle.setColor(veh_id, (1,1,0))
                    #print(counter)
                    if route_id == "S_E_N":
                        traci.vehicle.setRouteID(veh_id,'S_N')
                    if route_id == "S_E_W":
                        traci.vehicle.setRouteID(veh_id,'S_W')
                    if route_id == "E_W_N":
                        traci.vehicle.setRouteID(veh_id,'E_N')
                    if route_id == "E_N_S":
                        traci.vehicle.setRouteID(veh_id,'E_S')
                    if route_id == "N_E_S":
                        traci.vehicle.setRouteID(veh_id,'N_S')

            if route_id in ["S_W_E" , "N_S_E"]:
                #print("hello W2TL")#W2TL
                #print(traci.lanearea.getJamLengthMeters('W_E_1'),traci.lanearea.getJamLengthVehicle('W_E_1'))
                if traci.lanearea.getJamLengthMeters('W_E_1')>=40 or traci.lanearea.getJamLengthMeters('W_E_2')>=40 or traci.lanearea.getJamLengthMeters('W_E_3')>=40 or traci.lanearea.getJamLengthMeters('W_E_4')>=40:
                    traci.vehicle.setColor(veh_id, (1,1,0))
                    #print(counter)
                    if route_id == "S_W_E":
                        traci.vehicle.setRouteID(veh_id,'S_E')
                    if route_id == "N_S_E":
                        traci.vehicle.setRouteID(veh_id,'N_E')

        
                    
                    
                
                
            
        
     
        # if traci.lanearea.getJamLengthMeters('det_0')>=9 and traci.lanearea.getJamLengthMeters('det_1') >=9:
        #     if checker==True:
                
        #         traci.vehicle.setRouteID('carflow.10' , 'route_0')
        #         #traci.vehicle.setRouteID('carflow.11' , 'route_0')
        #         traci.vehicle.setRouteID('carflow.12' , 'route_0')
        #       #  traci.vehicle.setRouteID('carflow.13' , 'route_0')
        #         traci.vehicle.setRouteID('carflow.14' , 'route_0')
        #         checker=False
            
           
        

        
      
        #step += 1

    traci.close()
    sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "tlcs_config_train.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()
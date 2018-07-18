import quadcopter,gui,controller,q_controller
import signal
import sys
import argparse
import time

# Constants
TIME_SCALING = 0.0 # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002 # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005 # seconds
global run

def Single_AI(gui_on, get_motor_speeds):
    # Set goals to go to
    GOALS = [(1,1,5)]
    YAWS = [0]
    rand0 = random.rand()/10
    rand1 = random.rand()/10
    rand2 = random.rand()/10
    # Define the quadcopters
    QUADCOPTER={'q1':{'position':[1,0,4],'orientation':[rand0,rand1,rand2],'L':0.3,'r':0.1,'prop_size':[10,4.5],'weight':1.2}}
    # Controller parameters
    CONTROLLER_PARAMETERS = {'Motor_limits':[4000,9000],
                        'Tilt_limits':[-10,10],
                        'Yaw_Control_Limits':[-900,900],
                        'Z_XY_offset':500,
                        'Linear_PID':{'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
                        'Linear_To_Angular_Scaler':[1,1,0],
                        'Yaw_Rate_Scaler':0.18,
                        'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]},
                        }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    if gui_on:
        gui_object = gui.GUI(quads=QUADCOPTER)
    ctrl = controller.Controller_AI(quad.get_state,quad.get_time,quad.set_motor_speeds,params=CONTROLLER_PARAMETERS,quad_identifier='q1',get_motor_speeds=get_motor_speeds)
    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    # Update the GUI while switching between destination poitions
    global run
    while(run==True):
        #ctrl.update_target(GOALS[0])
        #ctrl.update_yaw_target(YAWS[0])
        if gui_on:
            gui_object.quads['q1']['position'] = quad.get_position('q1')
            gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
            gui_object.update()
        # Once quad gets below height of 2, kills simulation
        if quad.get_state('q1')[2] < 1:
            print(quad.get_orientation('q1'))
            run = False
    quad.stop_thread()
    ctrl.stop_thread()

def parse_args():
    parser = argparse.ArgumentParser(description="Quadcopter Simulator")
    parser.add_argument("--sim", help='currently only single_ai', default='single_ai')
    parser.add_argument("--time_scale", type=float, default=-1.0, help='Time scaling factor. 0.0:fastest,1.0:realtime,>1:slow, ex: --time_scale 0.1')
    parser.add_argument("--quad_update_time", type=float, default=0.0, help='delta time for quadcopter dynamics update(seconds), ex: --quad_update_time 0.002')
    parser.add_argument("--controller_update_time", type=float, default=0.0, help='delta time for controller update(seconds), ex: --controller_update_time 0.005')
    return parser.parse_args()

def signal_handler(signal, frame):
    global run
    run = False
    print('Stopping')
    sys.exit(0)

if __name__ == "__main__":
    args = parse_args()
    if args.time_scale>=0: TIME_SCALING = args.time_scale
    if args.quad_update_time>0: QUAD_DYNAMICS_UPDATE = args.quad_update_time
    if args.controller_update_time>0: CONTROLLER_DYNAMICS_UPDATE = args.controller_update_time
    AI = q_controller.Q_Controller()
    gui_on = False
    if args.sim == 'single_ai':
        global run
        for x in range(1000):
            run = True
            print("starting simulation: ", x)
            if x == 99:
                gui_on = True
            Single_AI(gui_on, AI.get_motor_speeds)
            print("simulation finished, starting training")
            AI.replay()

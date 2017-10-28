import numpy as np
import math
import scipy.integrate
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import time
import datetime
import threading
import signal
import sys

# Constants
TIME_SCALING = 1.0 # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002 # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005 # seconds

# Define quadcopter
QUADCOPTER={'position':[0,0,4],'orientation':[0,0,0],'L':0.2,'r':0.15,'prop_size':[10,4.5],'weight':1.2}
# Controller parameters
CONTROLLER_PARAMETERS = {'Motor_limits':[4000,9000],
                    'Tilt_limits':[-10,10],
                    'Z_XY_offset':500,
                    'Linear_PID':{'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
                    'Linear_To_Angular_Scaler':[1,1,0], 
                    'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,0.01],'D':[12000,12000,1800]},
                    }
# Set goals to go to
GOALS = [(1,1,2),(1,-1,4),(-1,-1,2),(-1,1,4)]

# Global variable
run = True

class Propeller():
    def __init__(self, prop_dia, prop_pitch, thrust_unit='N'):
        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.speed = 0 #RPM
        self.thrust = 0

    def set_speed(self,speed):
        self.speed = speed
        # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        self.thrust = 4.392e-8 * self.speed * math.pow(self.dia,3.5)/(math.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)
        if self.thrust_unit == 'Kg':
            self.thrust = self.thrust*0.101972

class Quadcopter():
    # State space representation: [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
    # From Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky
    def __init__(self,quadcopter_definition,gravity=9.81,b=0.0245):
        self.L = quadcopter_definition['L']
        self.state = np.zeros(12)
        self.m = quadcopter_definition['weight']
        self.g = gravity
        self.r = quadcopter_definition['r']
        self.b = b
        self.m1 = Propeller(quadcopter_definition['prop_size'][0],quadcopter_definition['prop_size'][1])
        self.m2 = Propeller(quadcopter_definition['prop_size'][0],quadcopter_definition['prop_size'][1])
        self.m3 = Propeller(quadcopter_definition['prop_size'][0],quadcopter_definition['prop_size'][1])
        self.m4 = Propeller(quadcopter_definition['prop_size'][0],quadcopter_definition['prop_size'][1])
        # From Quadrotor Dynamics and Control by Randal Beard
        ixx=((2*self.m*self.r**2)/5)+(2*self.m*self.L**2)
        iyy=ixx
        izz=((2*self.m*self.r**2)/5)+(4*self.m*self.L**2)
        self.I = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self.invI = np.linalg.inv(self.I)
        self.ode =  scipy.integrate.ode(self.state_dot)
        self.ode.set_integrator('vode',nsteps=500,method='bdf')
        self.time = datetime.datetime.now()
        self.thread_object = None
        self.set_position(quadcopter_definition['position'])
        self.set_orientation(quadcopter_definition['orientation'])

    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def state_dot(self, state, time):
        state_dot = np.zeros(12)
        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self.state[3]
        state_dot[1] = self.state[4]
        state_dot[2] = self.state[5]
        # The acceleration
        x_dotdot = np.array([0,0,-self.m*self.g]) + np.dot(self.rotation_matrix(self.state[6:9]),np.array([0,0,(self.m1.thrust+self.m2.thrust+self.m3.thrust+self.m4.thrust)]))/self.m
        state_dot[3] = x_dotdot[0]
        state_dot[4] = x_dotdot[1]
        state_dot[5] = x_dotdot[2]
        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = self.state[9]
        state_dot[7] = self.state[10]
        state_dot[8] = self.state[11]
        # The angular accelerations
        omega = self.state[9:12]
        tau = np.array([self.L*(self.m1.thrust-self.m3.thrust), self.L*(self.m2.thrust-self.m4.thrust), self.b*(self.m1.thrust-self.m2.thrust+self.m3.thrust-self.m4.thrust)])
        omega_dot = np.dot(self.invI, (tau - np.cross(omega, np.dot(self.I,omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot

    def update(self, dt):
        #self.state = scipy.integrate.odeint(self.state_dot, self.state, [0,dt])[1]
        self.ode.set_initial_value(self.state,0)
        self.state = self.ode.integrate(self.ode.t + dt)
        self.state[2] = max(0,self.state[2])

    def set_motor_speeds(self,speeds):
        self.m1.set_speed(speeds[0])
        self.m2.set_speed(speeds[1])
        self.m3.set_speed(speeds[2])
        self.m4.set_speed(speeds[3])

    def get_orientation(self):
        return self.state[6:9]

    def get_position(self):
        return self.state[0:3]

    def get_angular_rate(self):
        return self.state[9:12]

    def get_linear_rate(self):
        return self.state[3:6]

    def set_position(self,position):
        self.state[0:3] = position

    def set_orientation(self,orientation):
        self.state[6:9] = orientation

    def thread_run(self,dt,time_scaling):
        rate = time_scaling*dt
        last_update = self.time
        while(run==True):
            time.sleep(0)
            self.time = datetime.datetime.now()
            if (self.time-last_update).total_seconds() > rate:
                self.update(dt)
                last_update = self.time

    def start_thread(self,dt=0.002,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(dt,time_scaling))
        self.thread_object.start()

class GUI():
    def __init__(self, quadcopter_definition, get_position=None, get_orientation=None):
        if get_position is not None:
            self.get_position = get_position
        if get_orientation is not None:
            self.get_orientation = get_orientation
        self.L = quadcopter_definition['L']
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-2.0, 2.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-2.0, 2.0])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([0, 5.0])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')
        self.init_plot()

    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def init_plot(self):
        self.l1, = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
        self.l2, = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)
        self.hub, = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)

    def update(self, position=None, orientation=None):
        if position is None:
            position = self.get_position()
        if orientation is None:
            orientation = self.get_orientation()
        R = self.rotation_matrix(orientation)
        points = np.array([ [-self.L,0,0], [self.L,0,0], [0,-self.L,0], [0,self.L,0], [0,0,0], [0,0,0] ]).T
        points = np.dot(R,points)
        points[0,:] += position[0]
        points[1,:] += position[1]
        points[2,:] += position[2]
        self.l1.set_data(points[0,0:2],points[1,0:2])
        self.l1.set_3d_properties(points[2,0:2])
        self.l2.set_data(points[0,2:4],points[1,2:4])
        self.l2.set_3d_properties(points[2,2:4])
        self.hub.set_data(points[0,5],points[1,5])
        self.hub.set_3d_properties(points[2,5])
        plt.pause(0.000000000000001)

class Controller():
    def __init__(self, quad, params):
        self.quad = quad
        self.MOTOR_LIMITS = params['Motor_limits']
        self.TILT_LIMITS = [(params['Tilt_limits'][0]/180.0)*3.14,(params['Tilt_limits'][1]/180.0)*3.14]
        self.Z_LIMITS = [self.MOTOR_LIMITS[0]+params['Z_XY_offset'],self.MOTOR_LIMITS[1]-params['Z_XY_offset']]
        self.LINEAR_P = params['Linear_PID']['P']
        self.LINEAR_I = params['Linear_PID']['I']
        self.LINEAR_D = params['Linear_PID']['D']
        self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
        self.ANGULAR_P = params['Angular_PID']['P']
        self.ANGULAR_I = params['Angular_PID']['I']
        self.ANGULAR_D = params['Angular_PID']['D']
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.thread_object = None
        self.target = [0,0,0]

    def constrain_list(self,quantities,c_min,c_max):
        for i in range(len(quantities)):
            quantities[i] = min(max(quantities[i],c_min),c_max)

    def constrain(self,quantity,c_min,c_max):
        return min(max(quantity,c_min),c_max)

    def update(self):
        [dest_x,dest_y,dest_z] = self.target
        [x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot] = self.quad.state
        x_error = dest_x-x
        y_error = dest_y-y
        z_error = dest_z-z
        self.xi_term += self.LINEAR_I[0]*x_error
        self.yi_term += self.LINEAR_I[1]*y_error
        self.zi_term += self.LINEAR_I[2]*z_error
        dest_x_dot = self.LINEAR_P[0]*(x_error) + self.LINEAR_D[0]*(-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1]*(y_error) + self.LINEAR_D[1]*(-y_dot) + self.yi_term
        dest_z_dot = self.LINEAR_P[2]*(z_error) + self.LINEAR_D[2]*(-z_dot) + self.zi_term
        throttle = self.constrain(dest_z_dot,self.Z_LIMITS[0],self.Z_LIMITS[1])
        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_x_dot*math.sin(gamma)-dest_y_dot*math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_x_dot*math.cos(gamma)+dest_y_dot*math.sin(gamma))
        dest_gamma = 0
        dest_theta,dest_phi,dest_gamma = self.constrain(dest_theta,self.TILT_LIMITS[0],self.TILT_LIMITS[1]),self.constrain(dest_phi,self.TILT_LIMITS[0],self.TILT_LIMITS[1]),self.constrain(dest_gamma,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
        theta_error = dest_theta-theta
        phi_error = dest_phi-phi
        gamma_error = dest_gamma-gamma
        self.thetai_term += self.ANGULAR_I[0]*theta_error
        self.phii_term += self.ANGULAR_I[1]*phi_error
        self.gammai_term += self.ANGULAR_I[2]*gamma_error
        x_val = self.ANGULAR_P[0]*(theta_error) + self.ANGULAR_D[0]*(-theta_dot) + self.thetai_term
        y_val = self.ANGULAR_P[1]*(phi_error) + self.ANGULAR_D[1]*(-phi_dot) + self.phii_term
        z_val = self.ANGULAR_P[2]*(gamma_error) + self.ANGULAR_D[2]*(-gamma_dot) + self.gammai_term
        m1 = throttle + x_val + z_val
        m2 = throttle + y_val - z_val
        m3 = throttle - x_val + z_val
        m4 = throttle - y_val - z_val
        M = [m1,m2,m3,m4]
        self.constrain_list(M,self.MOTOR_LIMITS[0],self.MOTOR_LIMITS[1])
        self.quad.set_motor_speeds(M)

    def update_target(self,target):
        self.target = target

    def thread_run(self,update_rate,time_scaling):
        update_rate = update_rate*time_scaling
        last_update = self.quad.time
        while(run==True):
            time.sleep(0)
            self.time = self.quad.time
            if (self.time - last_update).total_seconds() > update_rate:
                self.update()
                last_update = self.time

    def start_thread(self,update_rate=0.005,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(update_rate,time_scaling))
        self.thread_object.start()

def signal_handler(signal, frame):
    global run
    run = False
    print('Stopping')
    sys.exit(0)

def Run():
    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controller
    quad = Quadcopter(QUADCOPTER)
    gui = GUI(QUADCOPTER,get_position=quad.get_position,get_orientation=quad.get_orientation)
    ctrl = Controller(quad,params=CONTROLLER_PARAMETERS)
    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    # Update the GUI while switching between destination poitions
    while(run==True):
        for goal in GOALS:
            ctrl.update_target(goal)
            for i in range(150):
                gui.update()

if __name__ == "__main__":
    Run()

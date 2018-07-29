import numpy as np
import math
import time
import threading

class Controller_AI():
    def __init__(self, get_state, get_time, actuate_motors, params, quad_identifier, get_motor_speeds):
        self.get_motor_speeds = get_motor_speeds
        self.quad_identifier = quad_identifier
        self.actuate_motors = actuate_motors
        self.get_state = get_state
        self.get_time = get_time
        self.MOTOR_LIMITS = params['Motor_limits']
        self.TILT_LIMITS = [(params['Tilt_limits'][0]/180.0)*3.14,(params['Tilt_limits'][1]/180.0)*3.14]
        self.YAW_CONTROL_LIMITS = params['Yaw_Control_Limits']
        self.Z_LIMITS = [self.MOTOR_LIMITS[0]+params['Z_XY_offset'],self.MOTOR_LIMITS[1]-params['Z_XY_offset']]
        self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
        self.YAW_RATE_SCALER = params['Yaw_Rate_Scaler']
        self.thread_object = None
        self.target = [0,0,0]
        self.yaw_target = 0.0
        self.run = True

    def wrap_angle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def update(self):
        [dest_x,dest_y,dest_z] = self.target
        data = [x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot] = self.get_state(self.quad_identifier)
        reward = -(theta)**2 + -(phi)**2 + -(gamma)**2 + -(theta_dot)**2 + -(phi_dot)**2 + -(gamma_dot)**2
        #print(reward)
        action = self.get_motor_speeds(reward, data)
        range_motors = self.MOTOR_LIMITS[1] - self.MOTOR_LIMITS[0]
        action = np.array(action)
        action *= range_motors #/ 4
        action += self.MOTOR_LIMITS[0]
        [m1, m2, m3, m4] = action
        M = np.clip([m1,m2,m3,m4],self.MOTOR_LIMITS[0],self.MOTOR_LIMITS[1])
        self.actuate_motors(self.quad_identifier,M)

    def update_target(self,target):
        self.target = target

    def update_yaw_target(self,target):
        self.yaw_target = self.wrap_angle(target)

    def thread_run(self,update_rate,time_scaling):
        update_rate = update_rate*time_scaling
        last_update = self.get_time()
        while(self.run==True):
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > update_rate:
                self.update()
                last_update = self.time

    def start_thread(self,update_rate=0.005,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(update_rate,time_scaling),daemon=True)
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

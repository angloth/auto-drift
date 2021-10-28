import gym
from gym import spaces
import carla
import numpy as np
from math import sin, cos, pi, acos, radians, degrees, exp
import time
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt

# State constants
NUM_OBS = 9
MAX_VEL = 42
MAX_ACC = 10

# Reward Constants
R_OOB = -200

# Calc constants
PI_2 = 2*pi
RESET_CONTROL = carla.VehicleControl(1.0, -0.1, 0.0, False, False, False, 0)
RESET_MIN_SPEED = 1

class DriftEnv(gym.Env):
    metadata = {'render.modes': ['gui', 'none']}

    def __init__(self, world, min_R, max_R, drift_center, max_episode_iters=500):
        self._world = world

        self.min_R = min_R
        self.max_R = max_R
        self.R = (max_R + min_R) / 2
        self.eps_tol = max_R - self.R
        self._drift_center = drift_center

        # Open AI gym vars
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        # [eps, beta, yaw, v_x, v_y, a_x, a_y, delta, tau]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(NUM_OBS,))

        # Obs range
        self._obs_max_vals = np.array([self.eps_tol+2, pi, pi, MAX_VEL, MAX_VEL, MAX_ACC, MAX_ACC, 1, 1])
        
        self._eps = None
        self._beta = None
        self._velocity = None
        self._delta_phi = None
        self._reward = None

        self.max_episode_iters = max_episode_iters
        self.iters = 0

        self.global_tick = 0

        self.reset()

        self.ticks = []
        self.betas = []
        self.epsilons = []
        self.rewards = []
        self.velocities = []

        self.last_10_betas = []
        self.last_10_epsilons = []
        self.last_10_rewards = []
        self.last_10_velocities = []

        self.is_training = False

    def log(self):
        self.last_10_betas.append(self._beta)
        self.last_10_epsilons.append(self._eps)
        self.last_10_rewards.append(self._reward)
        self.last_10_velocities.append(self._velocity)

        if len(self.last_10_betas) == 10:
            self.ticks.append(self.global_tick)

            self.betas.append(mean(self.last_10_betas))
            self.epsilons.append(mean(self.last_10_epsilons))
            self.rewards.append(mean(self.last_10_rewards))
            self.velocities.append(mean(self.last_10_velocities))

            self.last_10_betas = []
            self.last_10_epsilons = []
            self.last_10_rewards = []
            self.last_10_velocities = []

    def plot_logs(self):
        vals = [self.betas, self.epsilons, self.rewards, self.velocities]
        names = ["Beta", "Epsilon", "Reward", "Velocity"]
        print("HErer")

        for i in range(4):
            t = self.ticks
            s = vals[i]
            name = names[i]

            print("fck")
            fig, ax = plt.subplots()
            print("hsdfs")

            ax.plot(t, s)

            ax.set(xlabel='Timestep', title=name)
            print("aids")
            ax.grid()

            print("uhoh")

            fig.savefig(name + ".png")
            #plt.show()

    def save_logs(self):
        d = {'tick': self.ticks, 'beta': self.betas, 'eps': self.epsilons, 'rewards': self.rewards, 'vels': self.velocities}
        df = pd.DataFrame(data=d)

        df.to_csv("data.csv")

    def step(self, action):
        ## Apply action
        self._apply_action(action)

        ## Extract internal state
        obs = self._extract_obs()

        ## Calculate reward
        reward = self._calc_reward()
        self._reward = reward

        ## Calculate done
        done = self._calc_done()

        ## Calculate info
        info = {}

        self.iters += 1
        self.global_tick += 1
        
        if self.is_training:
            self.log()

        return obs, reward, done, info


    def _apply_action(self, action):
        delta, acc = action

       # print("delta:", delta, "acc:", acc)
        control = carla.VehicleControl()

        control.steer = float(delta)

        #control.throttle = float(acc)/2.5 + 0.6
        control.throttle = float(acc)/4 + 0.75
        
        control.hand_brake = False
        control.manual_gear_shift = False

        self._world.player.apply_control(control)

        #time.sleep(0.01)
        self._world.world.tick()


    def _extract_obs(self, debug=True):
        carla_pos = self._world.player.get_location()
        carla_vel = self._world.player.get_velocity()
        carla_acc = self._world.player.get_acceleration()
        carla_rot = self._world.player.get_transform().rotation

        pos_global = np.array([carla_pos.x, carla_pos.y])
        vel_global = np.array([carla_vel.x, carla_vel.y])
        acc_global = np.array([carla_acc.x, carla_acc.y])
        yaw_global = radians(carla_rot.yaw)


        # Extract alpha
        rel_pos_global = pos_global - self._drift_center
        alpha = np.arctan2(rel_pos_global[0], rel_pos_global[1])

        # Extract coordinates of new origin (collision point between circle and vector out from drift center)
        new_origin_global = np.array([sin(alpha)*self.R, cos(alpha)*self.R]) + self._drift_center

        if debug:
            min_R_point = carla.Location(self._drift_center[0] + self.min_R*sin(alpha), self._drift_center[1] +  self.min_R*cos(alpha), carla_pos.z+1)
            max_R_point = carla.Location(self._drift_center[0] + self.max_R*sin(alpha), self._drift_center[1] +  self.max_R*cos(alpha), carla_pos.z+1)

            drift_center_loc_1 = carla.Location(self._drift_center[0], self._drift_center[1], carla_pos.z)
            drift_center_loc_2 = carla.Location(self._drift_center[0], self._drift_center[1], carla_pos.z+15)
            self._world.world.debug.draw_line(drift_center_loc_1, drift_center_loc_2, life_time=1)
            
            self._world.world.debug.draw_point(min_R_point, life_time=50)
            
            self._world.world.debug.draw_point(max_R_point, life_time=50)

            
            self._world.world.debug.draw_point(carla.Location(new_origin_global[0], new_origin_global[1], carla_pos.z+1), life_time=50)

        # Extract coordinates in new system
        pos_new = pos_global - new_origin_global
        pos_new = rot_vec(pos_new, alpha)

        # Extract epsilon
        eps = pos_new[1]
        self._eps = eps

        # Extract beta
        yaw_new = yaw_global - alpha
        
        v_dir_global = np.arctan2(vel_global[0], vel_global[1])

        orientation = np.array([cos(yaw_global), sin(yaw_global)])

        dot_prod = 1
        if np.linalg.norm(vel_global) > 0.01:
            dot_prod = np.dot(orientation, vel_global / np.linalg.norm(vel_global))

            if(dot_prod > 1):
                dot_prod = 1
            if(dot_prod < -1):
                dot_prod = -1

        beta = acos(dot_prod)

        if np.cross(orientation, vel_global) > 0:
            beta = -beta
        
        # Extract yaw
        orientation_local = rot_vec(orientation, alpha)
        yaw_new = np.arctan2(orientation_local[0], orientation_local[1]) - pi/2

        # Extract v_x, v_y
        vel_new = rot_vec(vel_global, alpha)
        v_x, v_y = vel_new

        # Extract a_x, a_y
        acc_new = rot_vec(acc_global, alpha)
        a_x, a_y = acc_new

        # Get actions
        last_control = self._world.player.get_control()
        delta = last_control.steer
        throttle = last_control.throttle
        #tau = max((throttle-0.6) * 2.5, -1)
        tau = max((throttle-0.75) * 4, -1)

        # Normalize observations to [-1, 1]
        obs = np.array([eps, beta, yaw_new, v_x, v_y, a_x, a_y, delta, tau])
        #print(obs)

        norm_obs = np.array([obs[i] / self._obs_max_vals[i] for i in range(NUM_OBS)])
        #print("ORIG:", [x for x in obs])
        #print("NORM:", [x for x in norm_obs])

        ## Save values for easier reward calcs
        self._beta = degrees(beta)
        self._velocity = np.linalg.norm(vel_global)
        self._delta_phi = degrees(pi/2 - np.arctan2(vel_new[0], vel_new[1]))

        return norm_obs


    def _calc_reward(self):
        if abs(self._eps) > self.eps_tol:
            return R_OOB    
        else:
            return r_epsilon(self._eps) + 15*r_velocity(self._velocity) + r_slip(self._beta) + 1*r_phi(self._delta_phi)

    def _calc_done(self):
        return abs(self._eps) > self.eps_tol or self.iters > self.max_episode_iters

    def reset(self):
        
        self.iters = 0

        self._world.reset_player()

        while True:
            
            self._world.player.apply_control(RESET_CONTROL)
            self._world.world.tick()
    
            carla_vel = self._world.player.get_velocity()
            np_vel = np.array([carla_vel.x, carla_vel.y])

            if np.linalg.norm(np_vel) > RESET_MIN_SPEED:
                break
        
        return self._extract_obs()


    def render(self, mode='none', close=False):
        if mode == 'none':
            pass
        elif mode == 'gui':
            # TODO show GUI
            pass
    

def rot_vec(v, a):
    return np.array([v[0]*cos(a) - v[1]*sin(a), v[0]*sin(a) + v[1]*cos(a)])

def signed_angle_difference(a1, a2) -> float:
    """Calculates the signed angle difference in radians between a1 and a2, where a2 is the wanted angle
        Taken from: https://math.stackexchange.com/questions/1649841/signed-angle-difference-without-conditions"""
    return ((a1 - a2  + 3*pi) % PI_2) - pi 

    
def r_slip(beta):
    return 8*exp(-0.02*abs(beta-50)) - 2.955
    #return 1 / exp(-0.01 * abs(beta)) - 1

def r_velocity(velocity):
    return 1 / exp(-0.025 * velocity) - 1

def r_phi(delta_phi):
    return exp(-0.06 * abs(delta_phi))

def r_epsilon(epsilon):
    return exp(-0.9 * abs(epsilon))
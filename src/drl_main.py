import argparse
import glob
import logging
import os
import numpy as np
import numpy.random as random
import sys
import math
#import pygame

from stable_baselines3 import SAC

from world import World
from drl.environment import DriftEnv, rot_vec

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc


def main():

    #pygame.init()
   # pygame.font.init()

    world = None

    args = client_args()

    drift_env = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(1.0)

        # load world 
        map_name = 'Town03_Opt'
        if client.get_world().get_map().name != 'Carla/Maps/' + map_name:
            client.set_timeout(20.0)
            client.load_world(map_name)

        #   client.get_world().unload_map_layer(carla.MapLayer.All)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        # set synchronous mode 
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        # open pygame window
        #display = pygame.display.set_mode(
       #     (args.width, args.height),
       #     pygame.HWSURFACE | pygame.DOUBLEBUF)

        # Vehicle init vals
        start_location = carla.Location(3.852000713348389, 20.366968154907227, 0.0)
        start_rotation = carla.Rotation(0.0, -15.289385795593262, 0.0)
        start_transform = carla.Transform(start_location, start_rotation)

        # initialize HUD
        #hud = HUD(args.width, args.height)
        hud = None
        world = World(client.get_world(), hud, args, player_start_transform=start_transform)
        #controller = KeyboardControl(world)

        #clock = pygame.time.Clock()

        # Camera transform
        cam_location = carla.Location(-0.1046655997633934, -5.900334358215332, 49.59463882446289)
        cam_rotation = carla.Rotation(-83.52818298339844, 91.57373046875, -0.0030753209721297026)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        observer_obj = world.world.get_spectator()
        observer_obj.set_transform(cam_transform)

        # Drift variables
        #drift_center = np.array([-1.7998074293136597, 0.028291527181863785])
        drift_center = np.array([-0.8, 0.05])
        min_R = 19
        max_R = 22
        R = (max_R - min_R) / 2

        # Create drift env
        drift_env = DriftEnv(world, min_R, max_R, drift_center)
        drift_env.is_training = True

        # Train model
        model = SAC("MlpPolicy", drift_env, verbose=1, device="cpu")

        #model = SAC.load("Drift_test_4", device="cpu")
        #model.set_env(drift_env)

        model.learn(total_timesteps=5e4)

        model.save("Drift_test_6")

       # drift_env.plot_logs()
        drift_env.save_logs()
        drift_env.is_training = False

        print("Done")


        obs = drift_env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = drift_env.step(action)

            if done:
                obs = drift_env.reset()

        
        while True:            

            #action = np.array([0.0, 0.0])
           # action = np.array([-0.1])

            obs, reward, done, info = drift_env.step(action)

            #obs_pos = world.world.get_spectator().get_location()
            #obs_pos_np = np.array([obs_pos.x, obs_pos.y])
            #print("R:", np.linalg.norm(obs_pos_np - drift_center))

            #obs_transform = world.world.get_spectator().get_transform()
            #obs_transl = obs_transform.location
            #print("Cam transl:", [obs_transl.x, obs_transl.y, obs_transl.z])
            #obs_rot = obs_transform.rotation
            #print("Cam rot:", [obs_rot.pitch, obs_rot.yaw, obs_rot.roll])

            if done:
                if args.loop:
                    drift_env.reset()
                else:
                    break
        """ """      


    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()


class client_args():
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 2000

        self.width = 1280
        self.height = 720

        self.vehicle = "vehicle.tesla.model3"
        self.loop = True

if __name__ == "__main__":
    main()

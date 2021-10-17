from __future__ import print_function

import argparse
import glob
import logging
import os
import numpy.random as random
import sys

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


# ==============================================================================
# -- Import our code -----------------------------------------------------------
# ==============================================================================
from world import World
from hud import HUD, KeyboardControl

from navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from navigation.basic_agent import BasicAgent  # pylint: disable=import-error

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

#try:
#    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#except IndexError:
#    pass

#from navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
#from navigation.basic_agent import BasicAgent  # pylint: disable=import-error

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

class client_args():
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 2000

        self.width = 1280
        self.height = 720

        self.vehicle = "vehicle.tesla.model3"
        self.loop = True

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)

        # load world 
        client.load_world('Town01_opt')
        client.set_timeout(6.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        # set synchronous mode 
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        # open pygame window
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        # initialize HUD
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # create agent and set destination
        agent = BehaviorAgent(world.player, behavior="aggressive")
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)

        clock = pygame.time.Clock()

        while True:
            clock.tick()

            world.world.tick()
            #world.world.wait_for_tick()
            
            if controller.parse_events():
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("The target has been reached, searching for another target", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)

    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """Main method"""

    args = client_args()

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()
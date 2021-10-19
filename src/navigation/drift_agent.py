import numpy as np
import carla

from navigation.global_route_planner import GlobalRoutePlanner
from navigation.mpc_controller import ModelPredictiveController

class DriftAgent(object):

    def __init__(self, vehicle, mpc_controller_params, sampling_res=2):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._sampling_resolution = sampling_res

        # Initialize planners
        self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        self._local_planner = ModelPredictiveController(self._vehicle, mpc_controller_params)

    def _trace_route(self, start_waypoint, end_waypoint):
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location

        route, splinepath = self._global_planner.trace_route(start_location, end_location)
        return splinepath

    def set_destination(self, end_location, start_location=None, debug=True):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        start_location = self._vehicle.get_location()

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        splinepath = self._trace_route(start_waypoint, end_waypoint)


        # Draw splinepath on map
        if debug:
            N = 100
            s = np.linspace(0, splinepath.length, N)
            spline_x = splinepath.x(s)
            spline_y = splinepath.y(s)
            locs = [carla.Location(spline_x[i], spline_y[i], 0.5) for i in range(N)]
            draw_time = 200
            for i in range(1, N):
                self._world.debug.draw_line(locs[i-1], locs[i], thickness=0.2, 
                    life_time=draw_time, color=carla.Color(b=255))

        self._local_planner.set_global_plan(splinepath, end_location)

    def run_step(self, debug=False):
        """Execute one step of navigation."""

        control = self._local_planner.run_step(debug=debug)

        return control

    def done(self):
        return self._local_planner.done()
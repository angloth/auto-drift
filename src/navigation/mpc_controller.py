import math
import casadi
import numpy as np
import carla

class ModelPredictiveController():
    def __init__(self, vehicle, controller_params, path=None, goal_tol=1, dt=0.1):
        super().__init__()
        
        self._vehicle = vehicle
        self.past_steering = self._vehicle.get_control().steer

        self.end_location = None

        # Code from extra assignment
        self.plan = path
        self.gamma_d = controller_params['gamma_d']
        self.gamma_theta = controller_params['gamma_theta']
        self.gamma_u = controller_params['gamma_u']
        self.L = controller_params['L']
        self.steer_limit = controller_params['steer_limit']

        self.sample_rate = dt
        self.prediction_horizon = controller_params['h_p']
        self.N = int(self.prediction_horizon / dt)
        
        self.goal_tol = goal_tol
        self.d = []
        self.s0 = 0
        self.optimizer = None #self.construct_problem(), has to be done when path is set.

    def heading_error(self, theta, s):
        """Compute theta error
        Inputs
            theta - current heading angle
            s - projection point on path
            
        Outputs
            theta_e - heading error angle
        """
        h_s, nc_s = self.plan.heading(s)
        theta_s = np.arctan2(h_s[1], h_s[0])
        
        h = np.array([math.cos(theta), math.sin(theta)])
        
        cos_theta_e = np.dot(h_s, h)
        sin_theta_e = np.cross(h_s, h)
        
        theta_e = math.atan(sin_theta_e / cos_theta_e)
        
        return theta_e
        
    def construct_problem(self):
        """Formulate optimal control problem"""
        
        dt = self.sample_rate
        
        # Create an casadi.Opti instance.
        opti = casadi.Opti('conic')
        
        # Optimization parameters
        d0 = opti.parameter()
        th0 = opti.parameter()
        v = opti.parameter()
        curvature = opti.parameter(self.N)
        
        # Optimization variables
        X = opti.variable(2, self.N + 1)
        proj_error = X[0, :]
        head_error = X[1, :]
        
        # Control variable (steering angle)
        Delta = opti.variable(self.N)

        # Goal function we wish to minimize     
        J = self.gamma_d * casadi.sumsqr(proj_error) + self.gamma_theta * casadi.sumsqr(head_error) + \
            self.gamma_u * casadi.sumsqr(Delta)
        

        opti.minimize(J)
         
        # Simulate the system forwards using RK4 and the implemented error model.
        for k in range(self.N):
            k1 = self.error_model(X[:, k], v, Delta[k], curvature[k])
            k2 = self.error_model(X[:, k] + dt / 2 * k1, v, Delta[k], curvature[k])
            k3 = self.error_model(X[:, k] + dt / 2 * k2, v, Delta[k], curvature[k])
            k4 = self.error_model(X[:, k] + dt * k3, v, Delta[k], curvature[k])
            x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next)  
        
        # Problem constraints.
        opti.subject_to(proj_error[0] == d0)
        opti.subject_to(head_error[0] == th0)
        opti.subject_to(opti.bounded(-self.steer_limit, Delta, self.steer_limit))
        
        # The cost function is quadratic and the problem is linear by design,
        # this is utilized when choosing solver.
        # Other possible solvers provided by CasADi include: 'qpoases' and 'ipopt'...

        opts_dict = {
            "print_iter": False,
            "print_time": 0,
            "constr_viol_tol": 1e-12,
            # "max_iter": 100
        }
        
        opti.solver('qrqp', opts_dict)
        
        return opti.to_function('f', [d0, th0, v, curvature], [Delta])
        
    def error_model(self, w, v, delta, curvature):
        """Error model describing how the distance and heading error evolve for a certain input
            
        Input:
            w = (d, theta_e)
            v - velocity
            delta - input, steering angle
            curvature - current curvature
            
        Output:
            Casadi vector of Time derivative of d and theta_e
        """

        d_dot = v * w[1]
        theta_e_dot = v * (delta - curvature * (1 - curvature * w[0]))
        
        return casadi.vertcat(d_dot, theta_e_dot)

    def u(self, t, w):
        p_car = w[0:2]
        theta = w[2]
        v = w[3]
            
        # Compute d and theta_e errors as in the basic exercise state-feedback controller
        s, d = self.plan.project(p_car, self.s0)
        self.s0 = s
        
        theta_e = self.heading_error(theta, s)
        
        s_i = 0  # Position for start of prediction

        # Solve optimization problem over the prediction-horizon
        s_horizon = np.linspace(s_i, s_i + self.N * v * self.sample_rate, self.N)        
        Delta = self.optimizer(d, theta_e, v, self.plan.c(s_horizon))
                
        # Collect the controller output
        delta = float(Delta[0])
        acc = 0        
        self.d.append(d)

        return delta, acc

    def run_step(self, debug=False):
        transform = self._vehicle.get_transform()
        location = transform.location
        yaw = transform.rotation.yaw
        velocity = self._vehicle.get_velocity()
        
        vel_arr = np.array([velocity.x, velocity.y])
        
        w = [location.x, location.y, yaw, np.linalg.norm(vel_arr)]

        t = None

        delta, acc = self.u(t, w)

        acc = 0.5

        control = carla.VehicleControl()

        if acc >= 0.0:
            control.throttle = acc
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = abs(acc)
        
        if delta > self.past_steering + 0.1:
            delta = self.past_steering + 0.1
        elif delta < self.past_steering - 0.1:
            delta = self.past_steering - 0.1

        control.steer = delta / self.steer_limit
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = delta

        print(delta)

        return control

    def set_global_plan(self, splinepath, end_location):
        self.plan = splinepath
        self.optimizer = self.construct_problem()

        self.end_location = end_location
    
    def done(self):
        return self._vehicle.get_location().distance(self.end_location) < self.goal_tol
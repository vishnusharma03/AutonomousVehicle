#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np
import math

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self.current_index       = 0

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                self.current_index = min_idx  ## altered code
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('t_prev', 0.0)
        self.vars.create_var('error_term_prev', 0.0)
        self.vars.create_var('integral_sum_prev', 0.0)
        self.vars.create_var('throttle_previous', 0.0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            #### Custom Implementaion starts
            kp = 1.0
            ki = 0.2
            kd = 0.01
            u_max = 1
            u_min = 0

            throttle_output = 0
            brake_output    = 0

            st = t - self.vars.t_prev

            error_term = v_desired - v

            p = kp * error_term

            integral_sum = self.vars.integral_sum_prev + error_term * st

            i = ki * integral_sum

            # d = kd * ((error_term - self.vars.error_term_prev)/st)

            d = kd * ((v - self.vars.v_previous)/st)

            u = p+i+d

            if u > u_max:
                u_sat = u_max
                # do NOT update the integral: keep previous integral_sum
                integral_sum = self.vars.integral_sum_prev
            elif u < u_min:
                u_sat = u_min
                integral_sum = self.vars.integral_sum_prev
            else:
                # no saturation ⇒ safe to integrate
                integral_sum = self.vars.integral_sum_prev + error_term * st
                u_sat = u
            
            throttle_output = np.clip(u_sat, u_min, u_max)
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            #### Custom Implementaion starts
            steer_output = 0

            k = 0.3
            epsilon = 0.0001
            n = 20
            current_idx = self.current_index

            start_idx = max(0, current_idx - n)
            end_idx = min(len(self._waypoints)-1, current_idx + n)

            local_wp = self._waypoints[start_idx:end_idx+1]

            distances = [np.sqrt((x - wp[0])**2 + (y - wp[1])**2) for wp in local_wp]
            closest_idx = np.argmin(distances)
            closest_point = local_wp[closest_idx]

            if closest_idx == len(local_wp) - 1:
                x1, y1, _ = local_wp[closest_idx - 1]
                x2, y2, _ = local_wp[closest_idx]
            else:
                x1, y1, _ = local_wp[closest_idx]
                x2, y2, _ = local_wp[closest_idx + 1]

            yaw_path = math.atan2(y2-y1, x2-x1)

            yaw_diff_heading = yaw_path - yaw
            yaw_diff_heading = np.arctan2(np.sin(yaw_diff_heading), np.cos(yaw_diff_heading))

            # crosstrack error:
            cte = (x - closest_point[0])*(-1* np.sin(yaw_path)) + (y - closest_point[1])*(np.cos(yaw_path))

            # control law
            steer_angle = yaw_diff_heading - np.arctan((k*cte)/(epsilon + v))



            #### Custom Implementaion ends

            # Change the steer output with the lateral controller. 
            steer_output    = np.clip(steer_angle, -1.22, 1.22)


            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step
        self.vars.st_prev = t
        self.vars.error_term_prev = error_term
        self.vars.integral_sum_prev = integral_sum
        self.vars.throttle_previous = throttle_output   


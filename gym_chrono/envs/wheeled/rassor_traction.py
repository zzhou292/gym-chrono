# =======================================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2021 projectchrono.org
# All right reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =======================================================================================
# Authors: Jason Zhou
# =======================================================================================

#
# =======================================================================================
# =======================================================================================

# Chrono imports
import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.robot as robot
import random
import cmath
try:
    from pychrono import irrlicht as chronoirr
except:
    print('Could not import ChronoIrrlicht')
try:
    import pychrono.sensor as sens
except:
    print('Could not import Chrono Sensor')

try:
    from pychrono import irrlicht as chronoirr
except:
    print('Could not import ChronoIrrlicht')

from gym_chrono.envs.utils.perlin_bitmap_generator import generate_random_bitmap

# Gym chrono imports
# Custom imports
from gym_chrono.envs.ChronoBase import ChronoBaseEnv
from gym_chrono.envs.utils.utils import CalcInitialPose, chVector_to_npArray, npArray_to_chVector, SetChronoDataDirectories

# Standard Python imports
import os
import math
import numpy as np

# Gymnasium imports
import gymnasium as gym


class rassor_traction(ChronoBaseEnv):
    """
    Wrapper for the cobra chrono model into a gym environment.
    Mainly built for use with action space = 
    """

    def __init__(self, render_mode='human'):
        ChronoBaseEnv.__init__(self, render_mode)

        SetChronoDataDirectories()
        # Define action space 
        # Define the ranges for the different parts of the action space
        range_1 = (-np.pi/2, np.pi/2)  # range for the first 6 outputs
        range_2 = (-0.25, 0.25)        # range for the last 2 outputs
        self.action_space = gym.spaces.Box(
            low=np.array([range_1[0]] * 6 + [range_2[0]] * 2, dtype=np.float64),
            high=np.array([range_1[1]] * 6 + [range_2[1]] * 2, dtype=np.float64),
            shape=(8,),
            dtype=np.float64
        )

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=0.8, shape=(
                1, 60, 60), dtype=np.float64),
            "data": gym.spaces.Box(low=-10, high=10, shape=(14,), dtype=np.float64)})

        # -----------------------------
        # Chrono simulation parameters
        # -----------------------------
        self.system = None  # Chrono system set in reset method
        self.ground = None  # Ground body set in reset method
        self.rover = None  # Robot set in reset method
        self.driver = None  # Driver set in reset method
        self.terrain = None # Terrain set in reset method
        self.terrain_height = None # Height of the terrain at the rover's initial position
        self.goal = None  # Goal position set in reset method

        # progress tracking
        self._prev_dist = None

        # Frequncy in which we apply control
        self._control_frequency = 20
        # Dynamics timestep
        self._step_size = 1e-3
        self._sim_time = 0.0
        # Number of steps dynamics has to take before we apply control
        self._steps_per_control = round(
            1 / (self._step_size * self._control_frequency))

        # ---------------------------------
        # Gym Environment variables
        # ---------------------------------
        # Maximum simulation time (seconds)
        self._max_time = 15
        # Holds reward of the episode
        self.reward = 0
        self._debug_reward = 0
        # Observation of the environment
        self.observation = None
        # Flag to determine if the environment has terminated -> In the event of timeOut or reach goal
        self._terminated = False
        # Flag to determine if the environment has truncated -> In the event of a crash
        self._truncated = False
        # Flag to check if the render setup has been done -> Some problem if rendering is setup in reset
        self._render_setup = False

    def reset(self, seed=None, options=None):
        
        
        # -----------------------------
        # Set up system with collision
        # -----------------------------
        self.system = chrono.ChSystemSMC()
        self.system.Set_G_acc(chrono.ChVectorD(0, 0, -3.71))
        self.system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.001)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.001)

        # -----------------------------
        # Generate random SCM Terrain patch
        # Initialize the terrain using a bitmap for the height map
        bitmap_file = os.path.dirname(os.path.realpath(__file__)) + "/../data/terrain_bitmaps/height_map.bmp"

        generate_random_bitmap(shape=(252, 252), resolutions=[(4, 4)], mappings=[(-5.0, 5.0)], file_name=bitmap_file)


        # Create the SCM deformable terrain patch
        self.terrain = veh.SCMTerrain(self.system)
        self.terrain.SetSoilParameters(2e6,   # Bekker Kphi
                                0,     # Bekker Kc
                                1.1,   # Bekker n exponent
                                0,     # Mohr cohesive limit (Pa)
                                30,    # Mohr friction limit (degrees)
                                0.01,  # Janosi shear coefficient (m)
                                2e8,   # Elastic stiffness (Pa/m), before plastic yield
                                3e4    # Damping (Pa s/m), proportional to negative vertical speed (optional)
        )


        # Set plot type for SCM (false color plotting)
        self.terrain.SetPlotType(veh.SCMTerrain.PLOT_SINKAGE, 0, 0.1)

        # Initialize the SCM terrain, specifying the initial mesh grid
        self.terrain.Initialize(bitmap_file, 6, 6, 0.0, 0.8, 0.1)

        # Initialize terrain data
        self.terrain_height = np.zeros((60, 60))
        mesh_res = 0.1

        for x in range(60):
            for y in range(60):
                target_loc_x = mesh_res * x
                target_loc_y = mesh_res * y
                self.terrain_height[x, y] = self.terrain.GetInitHeight(chrono.ChVectorD(target_loc_x,target_loc_y,0.0))


        # -----------------------------
        # Create the Robot
        # -----------------------------
        self.driver = robot.RassorSpeedDriver(1.0)
        self.rover = robot.Rassor(self.system)
        self.rover.SetDriver(self.driver)
        self.rover.Initialize(chrono.ChFrameD(chrono.ChVectorD(-2.0, 0.0, self.terrain.GetInitHeight(chrono.ChVectorD(-3.0,0.0,0.0))+0.3), chrono.ChQuaternionD(1, 0, 0, 0)))


        # Optionally, enable moving patch feature (single patch around vehicle chassis)
        self.terrain.AddMovingPatch(self.rover.GetChassis().GetBody(), chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(5, 3, 1))


        # -----------------------------
        # Set goal point
        # -----------------------------

        def generate_random_point():
            while True:
                if random.choice([True, False]):
                    # x is within [-2.5, -2] or [2, 2.5], y is free between [-2.5, 2.5]
                    x = random.choice([random.uniform(-2.5, -2), random.uniform(2, 2.5)])
                    y = random.uniform(-2.5, 2.5)
                else:
                    # y is within [-2.5, -2] or [2, 2.5], x is free between [-2.5, 2.5]
                    y = random.choice([random.uniform(-2.5, -2), random.uniform(2, 2.5)])
                    x = random.uniform(-2.5, 2.5)

                # Check if the point is outside the bounding box [-3, -1.5] x [-2, 2]
                if not (-3 <= x <= -1.5 and -2 <= y <= 2):
                    break

            return x, y

        x_goal, y_goal = generate_random_point()

        self.goal = chrono.ChVectorD(x_goal, y_goal, self.terrain.GetInitHeight(chrono.ChVectorD(x_goal,y_goal,0.0)))

        goal_contact_material = chrono.ChMaterialSurfaceNSC()
        goal_mat = chrono.ChVisualMaterial()
        goal_mat.SetAmbientColor(chrono.ChColor(1., 0., 0.))
        goal_mat.SetDiffuseColor(chrono.ChColor(1., 0., 0.))

        goal_body = chrono.ChBodyEasySphere(0.1, 1000, True, False, goal_contact_material)

        goal_body.SetPos(self.goal)
        goal_body.SetBodyFixed(True)
        goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)

        self.system.Add(goal_body)

        # Wait for rover to settle
        self._sim_time = self.system.GetChTime()
        settle_time = 1.0
        while self._sim_time <= settle_time:
            self._sim_time = self.system.GetChTime()
            self.terrain.Synchronize(self._sim_time)
            self.rover.Update()
            self.terrain.Advance(self._step_size)
            self.system.DoStepDynamics(self._step_size)

        # -----------------------------
        # Get the intial observation
        # -----------------------------
        self.observation = self.get_observation()
        # _vector_to_goal is a chrono vector
        self._debug_reward = 0

        self._terminated = False
        self._truncated = False
        self._success = False

        return self.observation, {}

    def step(self, action):
        self._sim_time = self.system.GetChTime()

        self.driver.SetDriveMotorSpeed(0, action[0])
        self.driver.SetDriveMotorSpeed(1, action[1])
        self.driver.SetDriveMotorSpeed(2, action[2])
        self.driver.SetDriveMotorSpeed(3, action[3])

        self.driver.SetRazorMotorSpeed(0, action[4])
        self.driver.SetRazorMotorSpeed(1, action[5])

        # boundary check
        rot_0 = self.rover.GetArmMotorRot(0)
        rot_1 = self.rover.GetArmMotorRot(1)
        if rot_0 < -0.5 * np.pi or rot_0 > 0.5 * np.pi:
            action[6] = 0
        if rot_1 < -0.5 * np.pi or rot_1 > 0.5 * np.pi:
            action[7] = 0
        self.driver.SetArmMotorSpeed(0, action[6])
        self.driver.SetArmMotorSpeed(1, action[7])
    
        for i in range(self._steps_per_control):
            self._sim_time = self.system.GetChTime()
            self.terrain.Synchronize(self._sim_time)
            self.rover.Update()
            self.terrain.Advance(self._step_size)
            self.system.DoStepDynamics(self._step_size)

        # Get the observation
        self.observation = self.get_observation()
        # Get reward
        self.reward = self.get_reward()
        self._debug_reward += self.reward
        # Check if we are done
        self._is_terminated()
        self._is_truncated()

        return self.observation, self.reward, self._terminated, self._truncated, {}

    def render(self, mode='human'):
        """
        Render the environment
        """

        # ------------------------------------------------------
        # Add visualization - only if we want to see "human" POV
        # ------------------------------------------------------
        if mode == 'human':
            if self._render_setup == False:
                self.vis = chronoirr.ChVisualSystemIrrlicht()
                self.vis.AttachSystem(self.system)
                self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
                self.vis.SetWindowSize(1280, 720)
                self.vis.SetWindowTitle('Cobro RL playground')
                self.vis.Initialize()
                self.vis.AddSkyBox()
                self.vis.AddCamera(chrono.ChVectorD(
                    1.5, 5, 3), chrono.ChVectorD(0, 0, 0.5))
                self.vis.AddTypicalLights()
                self.vis.AddLightWithShadow(chrono.ChVectorD(
                    1.5, -50, 100), chrono.ChVectorD(0, 0, 0.5), 3, 4, 10, 40, 512)
                self._render_setup = True

            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        else:
            raise NotImplementedError

    def get_reward(self):
        reward = 0.0

        # Progress reward
        vector_to_goal = self.goal - self.rover.GetChassis().GetPos()
        dist = vector_to_goal.Length()

        if self._prev_dist is not None:
            reward += (self._prev_dist - dist) * 100
            self._prev_dist = dist
        else:
            self._prev_dist = dist
            reward += 0
        
        return reward

    def _is_terminated(self):
        if self._sim_time >= self._max_time:
            self.reward += -2000
            self._debug_reward += -2000
            self._terminated = True
            print("sim time out")
            print("debug reward: ", self._debug_reward)
            print("========================")

        if (self.goal - self.rover.GetChassis().GetPos()).Length() < 0.2:
            self.reward += (self._max_time - self.system.GetChTime()) * 1000
            self._debug_reward += (self._max_time - self.system.GetChTime()) * 1000
            self._terminated = True
            print("Goal reached")
            print("debug reward: ", self._debug_reward)
            print("========================")


    def _is_truncated(self):
        # out of terrain
        if self.rover.GetChassis().GetPos().x < -2.8 or self.rover.GetChassis().GetPos().x > 2.8 or \
        self.rover.GetChassis().GetPos().y < -2.8 or self.rover.GetChassis().GetPos().y > 2.8:
            self.reward += -1500
            self._debug_reward += -1500
            self._truncated = True
            print("Out of Terrain")
            print("debug reward: ", self._debug_reward)
            print("========================")
        



    def get_observation(self):
        data = np.zeros(14, dtype=np.float64)
        image = np.zeros((1, 60, 60), dtype=np.float64)

        image = self.terrain_height
        image = np.expand_dims(image, axis=0)

        data[0] = self.rover.GetChassis().GetPos().x
        data[1] = self.rover.GetChassis().GetPos().y
        data[2] = self.rover.GetChassis().GetPos().z

        data[3] = self.rover.GetChassisVel().x
        data[4] = self.rover.GetChassisVel().y
        data[5] = self.rover.GetChassisVel().z

        data[6] = self.rover.GetChassis().GetRot().e0
        data[7] = self.rover.GetChassis().GetRot().e1
        data[8] = self.rover.GetChassis().GetRot().e2
        data[9] = self.rover.GetChassis().GetRot().e3

        data[10] = self.rover.GetArmMotorRot(0)
        data[11] = self.rover.GetArmMotorRot(1)

        data[12] = self.goal.x
        data[13] = self.goal.y

        observation = {
            "image": image,
            "data": data
        }
        # For not just the priveledged position of the robot
        return observation

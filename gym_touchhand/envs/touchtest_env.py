import os
import numpy as np

from gym import utils, spaces
from gym.envs.robotics.hand import manipulate_touch_sensors

from gymTouch.touch import DiscreteTouch
from gymTouch.sensorpoints import plot_points

# Ensure we get the path separator correct on windows
MANIPULATE_BLOCK_XML = os.path.join("hand", "manipulate_block_touch_sensors.xml")
MANIPULATE_EGG_XML = os.path.join("hand", "manipulate_egg_touch_sensors.xml")
MANIPULATE_PEN_XML = os.path.join("hand", "manipulate_pen_touch_sensors.xml")


class ManipulateTouchSensorsTestEnv(manipulate_touch_sensors.ManipulateTouchSensorsEnv):
    def __init__(
        self,
        model_path,
        target_position,
        target_rotation,
        target_position_range,
        reward_type,
        initial_qpos={},
        randomize_initial_position=True,
        randomize_initial_rotation=True,
        distance_threshold=0.01,
        rotation_threshold=0.1,
        n_substeps=20,
        relative_control=False,
        ignore_z_target_rotation=False,
        touch_visualisation="on_touch",
        touch_get_obs="sensordata",
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "always": always shows touch sensor sites
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation

        """
        manipulate_touch_sensors.ManipulateTouchSensorsEnv.__init__(
            self,
            model_path,
            target_position,
            target_rotation,
            target_position_range,
            reward_type,
            initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position,
            randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold,
            rotation_threshold=rotation_threshold,
            n_substeps=n_substeps,
            relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
            touch_visualisation=touch_visualisation,
            touch_get_obs=touch_get_obs
        )

        # Attach touch
        self.touch = DiscreteTouch(self)

        # Add bodies/geoms that will be be sensing
        TOUCH_BODY_FILTER = ["palm", "middle", "proximal", "knuckle"]
        TOUCH_RESOLUTION = .03
        FINE_TOUCH_FILTER = ["distal"]
        FINE_TOUCH_RESOLUTION = .01

        for body_id in range(self.sim.model.nbody):
            body_name = self.sim.model.body_id2name(body_id)
            if any(f_string in body_name for f_string in TOUCH_BODY_FILTER):
                n_sensors = self.touch.add_body(body_id=body_id, resolution=TOUCH_RESOLUTION)
                print("Added {} to {}".format(n_sensors, body_name))
            if any(f_string in body_name for f_string in FINE_TOUCH_FILTER):
                n_sensors = self.touch.add_body(body_id=body_id, resolution=FINE_TOUCH_RESOLUTION)
                print("Added {} to {}".format(n_sensors, body_name))

        print("A total of {} sensors added to model".format(self.touch.get_total_sensor_count()))

        # Plot points for every geom, just to check
        for geom_id in self.touch.sensing_geoms:
            plot_points(self.touch.sensor_positions[geom_id],
                        self.touch.plotting_limits[geom_id],
                        title=self.sim.model.geom_id2name(geom_id))

        # Add haptic observations to observation space
        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
                observation_haptic=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation_haptic"].shape, dtype="float32"
                ),
            )
        )

    def _get_obs(self):
        obs = super()._get_obs()
        # Add touch sensor output to observations
        # Try-except is there to avoid errors during initialization since OpenAi robotics envs call _get_obs during
        # init, when self.touch is not yet set.
        try:
            obs["observation_haptic"] = self.touch.get_force_vector_obs()
        except AttributeError:
            pass
        return obs

    def _render_callback(self):
        super()._render_callback()
        # Print out active contacts
        contacts = self.touch.get_contacts()
        print("Active Contacts:")
        for contact_id, geom_id, _ in contacts:
            contact = self.sim.data.contact[contact_id]
            other_geom = contact.geom2
            if contact.geom2 == geom_id:
                other_geom = contact.geom1
            geom_name = self.sim.model.geom_id2name(geom_id)
            other_geom_name = self.sim.model.geom_id2name(other_geom)
            rel_force = self.touch.get_force_relative(contact_id, geom_id)
            rel_pos = self.touch.get_contact_position_relative(contact_id, geom_id)
            print("Active contact between {} and {}! Forces {} at position {}".format(geom_name,
                                                                                      other_geom_name,
                                                                                      rel_force,
                                                                                      rel_pos))

#    def _get_normal_obs(self):
#        """ Dummy function to show some of the touch interface"""
#        contacts = self.touch.get_contacts()
#        # Get normal force and vector for every contact
#        forces = self.touch.get_empty_sensor_dict(1)
#        vectors = self.touch.get_empty_sensor_dict(3)
#        for contact_id, geom_id, nearest_sensor in contacts:
#            normal_force = self.touch.get_normal_force(contact_id, geom_id)
#            normal_vector = self.touch.get_normal_vector(contact_id, geom_id)
#            # Sum up vectors and forces for every contact on this sensor point
#            vectors[geom_id][nearest_sensor] = touch_utils.weighted_sum_vectors(normal_vector, normal_force,
#                                                                                forces[geom_id][nearest_sensor],
#                                                                                vectors[geom_id][nearest_sensor])
#            forces[geom_id][nearest_sensor] += normal_force
#        # Pack them into sensor array
#        force_array = self.touch.flatten_sensor_dict(forces)
#        vector_array = self.touch.flatten_sensor_dict(vectors)


class HandBlockTouchSensorsTestEnv(ManipulateTouchSensorsTestEnv, utils.EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        touch_get_obs="sensordata",
        reward_type="sparse",
    ):
        utils.EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type
        )
        ManipulateTouchSensorsTestEnv.__init__(
            self,
            model_path=MANIPULATE_BLOCK_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
        )


class HandEggTouchSensorsTestEnv(ManipulateTouchSensorsTestEnv, utils.EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        touch_get_obs="sensordata",
        reward_type="sparse",
    ):
        utils.EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type
        )
        ManipulateTouchSensorsTestEnv.__init__(
            self,
            model_path=MANIPULATE_EGG_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            touch_visualisation="off",
        )


class HandPenTouchSensorsTestEnv(ManipulateTouchSensorsTestEnv, utils.EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        touch_get_obs="sensordata",
        reward_type="sparse",
    ):
        utils.EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type
        )
        ManipulateTouchSensorsTestEnv.__init__(
            self,
            model_path=MANIPULATE_PEN_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False,
            reward_type=reward_type,
            ignore_z_target_rotation=True,
            distance_threshold=0.05,
        )

import os
import numpy as np

from gym import utils, spaces
from gym.envs.robotics.hand import manipulate

from gymTouch.touch import DiscreteTouch, scale_linear
from gymTouch.utils import plot_points

# Ensure we get the path separator correct on windows
MANIPULATE_BLOCK_XML = os.path.join("hand", "manipulate_block_touch_sensors.xml")
MANIPULATE_EGG_XML = os.path.join("hand", "manipulate_egg_touch_sensors.xml")
MANIPULATE_PEN_XML = os.path.join("hand", "manipulate_pen_touch_sensors.xml")


class ManipulateTouchSensorsTestEnv(manipulate.ManipulateEnv):
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
    ):
        # Pass through all the original env setup
        manipulate.ManipulateEnv.__init__(
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
                n_sensors = self.touch.add_body(body_id=body_id, scale=TOUCH_RESOLUTION)
                print("Added {} to {}".format(n_sensors, body_name))
            if any(f_string in body_name for f_string in FINE_TOUCH_FILTER):
                n_sensors = self.touch.add_body(body_id=body_id, scale=FINE_TOUCH_RESOLUTION)
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
            obs["observation_haptic"] = self.touch.get_touch_obs(DiscreteTouch.get_force_relative, 3, scale_linear)
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


class HandBlockTouchSensorsTestEnv(ManipulateTouchSensorsTestEnv, utils.EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        reward_type="sparse",
    ):
        utils.EzPickle.__init__(
            self, target_position, target_rotation, reward_type
        )
        ManipulateTouchSensorsTestEnv.__init__(
            self,
            model_path=MANIPULATE_BLOCK_XML,
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
        reward_type="sparse",
    ):
        utils.EzPickle.__init__(
            self, target_position, target_rotation, reward_type
        )
        ManipulateTouchSensorsTestEnv.__init__(
            self,
            model_path=MANIPULATE_EGG_XML,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
        )


class HandPenTouchSensorsTestEnv(ManipulateTouchSensorsTestEnv, utils.EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        reward_type="sparse",
    ):
        utils.EzPickle.__init__(
            self, target_position, target_rotation, reward_type
        )
        ManipulateTouchSensorsTestEnv.__init__(
            self,
            model_path=MANIPULATE_PEN_XML,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False,
            reward_type=reward_type,
            ignore_z_target_rotation=True,
            distance_threshold=0.05,
        )

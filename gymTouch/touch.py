import math
import numpy as np
import mujoco_py

from gymTouch.utils import mulRotT, mulRot
from gymTouch.sensorpoints import spread_points_box, spread_points_sphere, spread_points_cylinder, spread_points_capsule

# Class that handles all of this
#   Initialized as part of gym env, links to gym env
#   Function to add a sensing body/geom,
#   Adding a body that was already added overwrites existing sensor points
#   Figures out where to put sensor points
#   Stores sensor locations for each geom
#   Function to get normal vectors at sensor points
#   TODO: Handles observation space automatically (???)
#   Function to read out contacts
#   TODO: Convienience functions for cleanup of automatic sensor placement: Remove sensors located within another geom
#   TODO: Function for "slippage" at sensor point: Direction + magnitude


GEOM_TYPES = {"PLANE": 0, "HFIELD": 1, "SPHERE": 2, "CAPSULE": 3, "ELLIPSOID": 4, "CYLINDER": 5, "BOX": 6, "MESH": 7}


class DiscreteTouch:

    def __init__(self, env, verbose=False):
        """
        env should be an openAI gym environment using mujoco. Critically env should have an attribute sim which is a
        mujoco-py sim object

        Sensor positions is a dictionary where each key is the index of a geom with sensors and the corresponding value
        is a numpy array storing the sensor positions on that geom: {geom_id: ndarray((n_sensors, 3))}
        Sensor positions are in relative coordinates for the geom.

        The sensor position dictionary is populated when a body/geom is added and can be overwritten afterwards
        """
        self.env = env
        self.sensor_positions = {}
        self.plotting_limits = {}
        self.verbose = verbose
        
    def add_body(self, body_id: int = None, body_name: str = None, resolution: float = math.inf):
        """Adds sensors to all geoms belonging to the given body. Returns the number of sensor points added"""
        if body_id is None and body_name is None:
            raise RuntimeError("DiscreteTouch.add_body called without name or id")
            
        if body_id is None:
            body_id = self.env.sim.model.body_name2id(body_name)

        n_sensors = 0
        for geom_id in range(self.env.sim.model.ngeom):
            g_body_id = self.env.sim.model.geom_bodyid[geom_id]
            contype = self.env.sim.model.geom_contype[geom_id]
            # Add a geom if it belongs to body and has collisions enabled (at least potentially)
            if g_body_id == body_id and contype > 0:
                n_sensors += self.add_geom(geom_id=geom_id, resolution=resolution)
        return n_sensors

    def add_geom(self, geom_id: int = None, geom_name: str = None, resolution: float = math.inf):
        if geom_id is None and geom_name is None:
            raise RuntimeError("DiscreteTouch.add_geom called without name or id")
            
        if geom_id is None:
            geom_id = self.env.sim.model.geom_name2id(geom_name)

        if self.env.sim.model.geom_contype[geom_id] == 0:
            raise RuntimeWarning("Added sensors to geom with collisions disabled!")

        if self.verbose:
            print("Geom {} name {} type {} ".format(
                  geom_id,
                  self.env.sim.model.geom_id2name(geom_id),
                  self.env.sim.model.geom_type[geom_id]))

        return self._add_sensorpoints(geom_id, resolution)

    @property
    def sensing_geoms(self):
        """ Returns the ids of all geoms with sensors """
        return list(self.sensor_positions.keys())

    def has_sensors(self, geom_id):
        """ Returns true if the geom has sensors """
        return geom_id in self.sensor_positions

    def get_sensor_count(self, geom_id):
        """ Returns the number of sensors for the geom """
        return self.sensor_positions[geom_id].shape[0]

    def get_total_sensor_count(self):
        """ Returns the total number of haptic sensors in the model """
        n_sensors = 0
        for geom_id in self.sensing_geoms:
            n_sensors += self.get_sensor_count(geom_id)
        return n_sensors

    def _add_sensorpoints(self, geom_id: int, resolution: float):
        # Add sensor points for the given geom using given resolution
        # Returns the number of sensor points added
        # Also set the maximum size of the geom, for plotting purposes
        geom_type = self.env.sim.model.geom_type[geom_id]
        size = self.env.sim.model.geom_size[geom_id]
        limit = 1
        if geom_type == GEOM_TYPES["BOX"]:
            limit = np.max(size)
            points = spread_points_box(resolution, size)
        elif geom_type == GEOM_TYPES["SPHERE"]:
            limit = size[0]
            points = spread_points_sphere(resolution, size[0])
        elif geom_type == GEOM_TYPES["CAPSULE"]:
            limit = size[1] + size[0]
            points = spread_points_capsule(resolution, 2*size[1], size[0])
        elif geom_type == GEOM_TYPES["CYLINDER"]:
            # Cylinder size 0 is radius, size 1 is half length
            limit = np.max(size)
            points = spread_points_cylinder(resolution, 2*size[1], size[0])
        elif geom_type == GEOM_TYPES["PLANE"]:
            RuntimeWarning("Cannot add sensors to plane geoms!")
            return None
        elif geom_type == GEOM_TYPES["ELLIPSOID"]:
            raise NotImplementedError("Ellipsoids currently not implemented")
        elif geom_type == GEOM_TYPES["MESH"]:
            size = self.env.sim.model.geom_rbound[geom_id]
            limit = size
            points = spread_points_sphere(resolution, size)
        else:
            return None
        if self.verbose:
            print("Added {} points to geom".format(points.shape[0]))
        self.plotting_limits[geom_id] = limit
        self.sensor_positions[geom_id] = points
        return points.shape[0]

    def get_nearest_sensor(self, contact_id, geom_id):
        """ Given a contact and a geom, return the sensor on the geom closest to the contact.
        Returns the sensor index and the distance between contact and sensor"""
        relative_position = self.get_contact_position_relative(contact_id, geom_id)
        sensor_points = self.sensor_positions[geom_id]
        distances = np.linalg.norm(sensor_points - relative_position, axis=1)
        idx = np.argmin(distances)
        return idx, distances[idx]

# ======================== Positions and rotations ================================
# =================================================================================

    def get_geom_position(self, geom_id):
        """ Returns world position of geom"""
        return self.env.sim.data.geom_xpos[geom_id]

    def get_geom_rotation(self, geom_id):
        """ Returns rotation matrix of geom frame relative to world frame"""
        return np.reshape(self.env.sim.data.geom_xmat[geom_id], (3, 3))

    def world_pos_to_relative(self, position, geom_id):
        """ Converts a (3,) numpy array containing xyz coordinates in world frame to geom frame"""
        rel_pos = position - self.get_geom_position(geom_id)
        rel_pos = mulRot(rel_pos, self.get_geom_rotation(geom_id))
        return rel_pos

    def relative_pos_to_world(self, position, geom_id):
        """ Converts a (3,) numpy array containing xyz coordinates in geom frame to world frame"""
        global_pos = mulRotT(position, self.get_geom_rotation(geom_id))
        global_pos = global_pos + self.get_geom_position(geom_id)
        return global_pos

    def get_contact_position_world(self, contact_id):
        """ Get the position of a contact in world frame """
        return self.env.sim.data.contact[contact_id].pos

    def get_contact_position_relative(self, contact_id, geom_id: int):
        """ Get the position of a contact in the geom frame """
        return self.world_pos_to_relative(self.get_contact_position_world(contact_id), geom_id)

# =============== Various types of forces in various frames =======================
# =================================================================================

    def get_force(self, contact_id, geom_id):
        """ Returns the full contact force in mujocos own contact frame"""
        forces = np.zeros(6, dtype=np.float64)
        mujoco_py.functions.mj_contactForce(self.env.sim.model, self.env.sim.data, contact_id, forces)
        contact = self.env.sim.data.contact[contact_id]
        if geom_id == contact.geom1:
            forces *= -1  # Convention is that normal points away from geom1
        elif geom_id == contact.geom2:
            pass
        else:
            RuntimeError("Mismatch between contact and geom")
        return forces[:3]

    def get_normal_force(self, contact_id, geom_id):
        """ Returns the normal force as a scalar"""
        return self.get_force(contact_id, geom_id)[0]

    def get_normal_vector(self, contact_id, geom_id):
        """ Returns the normal vector (unit vector in direction of normal) in geom frame"""
        contact = self.env.sim.data.contact[contact_id]
        normal_vector = contact.frame[:3]
        if geom_id == contact.geom1:  # Mujoco vectors point away from geom1 by convention
            normal_vector *= -1
        elif geom_id == contact.geom2:
            pass
        else:
            RuntimeError("Mismatch between contact and geom")
        # contact frame is in global coordinate frame, rotate to geom frame
        normal_vector = mulRot(normal_vector, self.get_geom_rotation(geom_id))
        return normal_vector

    def get_force_world(self, contact_id, geom_id):
        """ Returns full contact force in world frame"""
        contact = self.env.sim.data.contact[contact_id]
        forces = self.get_force(contact_id, geom_id)
        force_rot = np.reshape(contact.frame, (3, 3))
        global_forces = mulRotT(forces, force_rot)
        return global_forces

    def get_force_relative(self, contact_id, geom_id):
        """ Returns full contact force in geom frame"""
        global_forces = self.get_force_world(contact_id, geom_id)
        relative_forces = mulRot(global_forces, self.get_geom_rotation(geom_id))
        return relative_forces

    def get_contacts(self, verbose: bool = None):
        """ Returns a tuple containing (contact_id, geom_id, sensor_id) for each active contact with a sensing geom,
        where contact_id is the index of the contact in the mujoco arrays, geom_id is the index of the geom in the
        arrays and sensor id is the index of the sensor on the geom nearest to the contact"""
        if verbose is None:
            verbose = self.verbose
        sim = self.env.sim
        contact_geom_tuples = []
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            # Do we sense this contact at all
            if self.has_sensors(contact.geom1) or self.has_sensors(contact.geom2):
                rel_geoms = []
                if self.has_sensors(contact.geom1):
                    rel_geoms.append(contact.geom1)
                if self.has_sensors(contact.geom2):
                    rel_geoms.append(contact.geom2)

                forces = self.get_force(i, contact.geom1)
                if abs(forces[0]) < 1e-8:  # Contact probably inactive
                    continue

                for rel_geom in rel_geoms:
                    rel_pos = self.get_contact_position_relative(i, rel_geom)
                    nearest_sensor_id, distance = self.get_nearest_sensor(i, rel_geom)
                    contact_geom_tuples.append((i, rel_geom, nearest_sensor_id))

                    if verbose:
                        print("Sensing with geom: ", rel_geom, sim.model.geom_id2name(rel_geom))
                        print("Relative position: ", rel_pos)
                        print("Nearest sensor {} at position {} with distance {}".format(
                              nearest_sensor_id,
                              self.sensor_positions[rel_geom][nearest_sensor_id],
                              distance))
        return contact_geom_tuples

    def get_empty_sensor_dict(self, size):
        """ Returns a dictionary with empty sensor outputs. Keys are geom ids, corresponding values are the output
        arrays. For every geom with sensors, returns an empty numpy array of shape (n_sensors, size)"""
        sensor_outputs = {}
        for geom_id in self.sensor_positions:
            sensor_outputs[geom_id] = np.zeros((self.get_sensor_count(geom_id), size), dtype=np.float32)
        return sensor_outputs

    def flatten_sensor_dict(self, sensor_dict):
        """ Flattens a sensor dict, such as from get_empty_sensor_dict into a single large array in a deterministic
        fashion. Geoms with lower id come earlier, sensor outputs correspond to sensor_positions"""
        sensor_arrays = []
        for geom_id in sorted(self.sensor_positions):
            sensor_arrays.append(sensor_dict[geom_id])
        return np.concatenate(sensor_arrays)

    def get_force_vector_obs(self):
        """ Does the full contact getting-processing process, such that we get the full force vector on each sensor
        point for every sensor in the model
        Returns a numpy array of shape (n_sensors_total, 3)"""
        contact_tuples = self.get_contacts(verbose=False)

        sensor_outputs = self.get_empty_sensor_dict(3)

        for contact_id, geom_id, nearest_sensor_id in contact_tuples:
            rel_forces = self.get_force_relative(contact_id, geom_id)
            sensor_outputs[geom_id][nearest_sensor_id] += rel_forces

        sensor_obs = self.flatten_sensor_dict(sensor_outputs)
        return sensor_obs

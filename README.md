# gymTouch
Simple implementation for adding haptic sensors to openAI gym environments using mujoco


## Usage

Adding touch sensors to an existing gym environment is very straightforward, an example can be seen the sample environment in ```gymTouchTestEnv```. Attach the touch library, add which bodies/geoms should have sensors and at what resolution and then decide how you want to handle the output.

```
from gymTouch.touch import DiscreteTouch

class envWithTouch(envWithoutTouch):
  
  def __init__(self, *args, **kwargs):
  
    # Init the original environment, passing through the arguments
    super().__init__(*args, **kwargs)
    self.touch = DiscreteTouch(self) # Add touch to environment
    
    
    # Add sensors to model. Resolution is the approximate distance between sensors. Actual distance may vary.
    self.touch.add_body(name="name-of-body", resolution=.1)
    self.touch.add_geom(name="name-of-geom", resolution=.1)
    
    # Sensors are handled by DiscreteTouch.sensor_positions. You can add sensors to a geom yourself given the geoms id. 
    # Sensor positions are stored as a numpy array of shape (n_sensors, 3). Coordinates are in the frame of the associated geom.
    self.touch.sensor_positions[geom_id] = np.array([[0., 0., 0.,],]) # Adding a sensor at the origin of the geom
    
    
    # Depending on the environment you will have to fix the observation space. In this example we assume a Dictionary space and just add another entry
    obs = self._get_obs()
      self.observation_space = spaces.Dict(
        dict(
          observation=spaces.Box(
            -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
          ),
          observation_haptic=spaces.Box(
            -np.inf, np.inf, shape=obs["observation_haptic"].shape, dtype="float32"
          ),
        )
      )
    
    
  # Return touch information by adding an extra entry to observation dictionary. This only works if the environment is using a Dictionary observation space!
  def _get_obs(self):
    obs = super()._get_obs()
    # Add touch sensor output to observations
    try:
      obs["observation_haptic"] = self.touch.get_force_vector_obs()
    except AttributeError:
      pass
    return obs
```

The only requirement is that the environment be a mujoco environment with an attribute ```sim```, as that attribute is used to access the underlying mujoco simulation.

In the example above we return touch information as a force vector. Look into ```get_force_vector_obs``` and the sample environment if you would like different types of touch information, such as the normal force only.

You can also plot the points for a given geom using ``` plot_points from gymTouch.utils```.

#### Sample Environment
There are three sample environments based on the openAI Manipulate gym environments. Sensors were added to the palm and all fingers with the finger tips using a higher resolution. The environments are ```TouchTestBlock-v0```, ```TouchTestEgg-v0``` and ```TouchTestPen-v0```

After installation the sample environments can be used like any other gym environment:

```
import gym
import gymTouchTestEnv

env = gym.make("TouchTestEgg-v0")
```

By default these show you the sensor points added to each geom during initialization and print touch information on render updates.


## Installation
This repository provides two packages, the touch library 'gymTouch' and a sample environment 'gym_touchhand' based on the manipulateHand envs from openAI gym.
To install the touch library
```
Install mujoco-py and the openAi robotics gym environments following their instructions
Clone this repository
Move into the repository
cd gymTouch
pip install -e .
```

To install the sample environment first install gymTouch, then
```
cd gymTouchTestEnv
pip install -e .
```

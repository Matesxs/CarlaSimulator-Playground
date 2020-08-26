import carla
from carla import ColorConverter as cc
import random
import math
import time
import numpy as np
import gc
import cv2

from helpers.logger import init_logger
import settings

logger = init_logger("Agent", False)

class Agent:
  def __init__(self, identifier:int, client, train:bool=True):
    self.client = client
    self._id = identifier

    self.world = self.client.get_world()
    self.blueprint_library = self.world.get_blueprint_library()
    self.vehicle_bp = self.blueprint_library.filter("model3")[0]

    self.camera_bp = self.blueprint_library.find("sensor.camera.rgb")
    self.camera_bp.set_attribute("image_size_x", f"{settings.CAMERA_IMAGE_DIMENSIONS[0]}")
    self.camera_bp.set_attribute("image_size_y", f"{settings.CAMERA_IMAGE_DIMENSIONS[1]}")
    self.camera_bp.set_attribute("fov", f"{settings.CAMERA_FOV}")

    self.p_camera_bp = self.blueprint_library.find("sensor.camera.rgb")
    self.p_camera_bp.set_attribute("image_size_x", f"{settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[0]}")
    self.p_camera_bp.set_attribute("image_size_y", f"{settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[1]}")
    self.p_camera_bp.set_attribute("fov", f"{settings.PREVIEW_CAMERA_FOV}")

    self.colsenzor_bp = self.blueprint_library.find("sensor.other.collision")

    self.front_transform = carla.Transform(carla.Location(x=2.3, z=1.22))
    self.p_camera_transform = carla.Transform(carla.Location(x=-5, z=2.2), carla.Rotation(pitch=-20))
    self.left_transform = carla.Transform(carla.Location(x=-0.6, z=1.22, y=-0.8), carla.Rotation(yaw=-90))
    self.right_transform = carla.Transform(carla.Location(x=-0.6, z=1.22, y=0.8), carla.Rotation(yaw=90))

    self.lights = carla.VehicleLightState.LowBeam
    self.lights |= carla.VehicleLightState.Position

    self.train = train

    self.initialized = False
    self.agent_vehicle = None
    self.collision_hist = []
    self.actor_list = []
    self.episode_start = None

    self.prev_camera = None
    self.cameras = {}

  def __collision_data(self, event):
    collision_actor_id = event.other_actor.type_id
    collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

    for actor_id, impulse in settings.COLLISION_FILTER:
      if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
        return

    self.collision_hist.append(event)

  def __process_image(self, data, dest:str=None):
    try:
      data.convert(cc.Raw)
      array = np.reshape(np.array(data.raw_data), (data.height, data.width, 4))

      array = array[:, :, :3]
      array = array[:, :, ::-1]

      if not dest:
        self.prev_camera = array
      else:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        self.cameras[dest] = array
    except MemoryError:
      gc.collect()
      logger.error("Memory low")

  def clear_agent(self):
    self.initialized = False

    for actor in self.actor_list:
      try:
        if hasattr(actor, 'is_listening'):
          if actor.is_listening:
            actor.stop()
        if actor.is_alive:
          actor.destroy()
      except:
        pass

    self.actor_list = []
    self.collision_hist = []
    self.agent_vehicle = None
    self.prev_camera = None
    self.cameras = {}

  def check_rotation(self):
    rotation = self.agent_vehicle.get_transform().rotation
    x = abs(rotation.roll)
    y = abs(rotation.pitch)

    if x > settings.ROTATION_THRESHOLD or y > settings.ROTATION_THRESHOLD:
      return False

    return True

  def get_speed(self):
    v = self.agent_vehicle.get_velocity()
    return int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

  def spawn(self):
    self.clear_agent()
    self.episode_start = None

    # Spawn main vehicle
    spawn_selected = False
    while not spawn_selected:
      try:
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.agent_vehicle = self.world.spawn_actor(self.vehicle_bp, transform)
        spawn_selected = True
      except Exception:
        pass

    self.agent_vehicle.set_light_state(carla.VehicleLightState(self.lights))
    self.actor_list.append(self.agent_vehicle)

    # Add preview camera
    prev_camera = self.world.spawn_actor(self.p_camera_bp, self.p_camera_transform, attach_to=self.agent_vehicle)
    prev_camera.listen(lambda data: self.__process_image(data))
    self.actor_list.append(prev_camera)

    # Add front camera
    front_camera = self.world.spawn_actor(self.camera_bp, self.front_transform, attach_to=self.agent_vehicle)
    front_camera.listen(lambda data: self.__process_image(data, "front"))
    self.actor_list.append(front_camera)

    # Add left camera
    left_camera = self.world.spawn_actor(self.camera_bp, self.left_transform, attach_to=self.agent_vehicle)
    left_camera.listen(lambda data: self.__process_image(data, "left"))
    self.actor_list.append(left_camera)

    # Add left camera
    right_camera = self.world.spawn_actor(self.camera_bp, self.right_transform, attach_to=self.agent_vehicle)
    right_camera.listen(lambda data: self.__process_image(data, "right"))
    self.actor_list.append(right_camera)

    self.agent_vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
    time.sleep(1)

    # Add colision sensor to vehicle
    col_sensor = self.world.spawn_actor(self.colsenzor_bp, self.front_transform, attach_to=self.agent_vehicle)
    col_sensor.listen(lambda event: self.__collision_data(event))
    self.actor_list.append(col_sensor)

    selected_default_action = settings.ACTIONS[settings.DEFAULT_ACTION]
    self.agent_vehicle.apply_control(carla.VehicleControl(throttle=selected_default_action[0] * settings.THROT_AM, steer=selected_default_action[1] * settings.STEER_AM, brake=selected_default_action[2] * settings.BRAKE_AM))
    time.sleep(0.125)

    while self.cameras["front"] is None:
      time.sleep(0.01)
    while self.cameras["left"] is None:
      time.sleep(0.01)
    while self.cameras["right"] is None:
      time.sleep(0.01)
    while self.prev_camera is None:
      time.sleep(0.01)

    time.sleep(0.5)

    if len(self.collision_hist) != 0:
      self.spawn()

    self.initialized = True
    self.episode_start = time.time()
    return [self.cameras["front"], self.cameras["left"], self.cameras["right"], self.get_speed(), settings.DEFAULT_ACTION]

  def step(self, action:int):
    done = False

    kmh = self.get_speed()

    if len(self.collision_hist) != 0:
      done = True
      reward = settings.COLLISION_REWARD
    else:
      reward = kmh * (settings.SPEED_MAX_REWARD - settings.SPEED_MIN_REWARD) / 100 + settings.SPEED_MIN_REWARD

      if self.episode_start + settings.SECONDS_PER_EXPISODE < time.time() and self.train:
        done = True

      if not self.check_rotation():
        done = True
        reward = settings.BAD_ROTATION_REWARD

      if settings.TIME_WEIGHTED_REWARDS:
        reward *= (time.time() - self.episode_start) / settings.SECONDS_PER_EXPISODE

      selected_action = settings.ACTIONS[action]
      if selected_action:
        self.agent_vehicle.apply_control(carla.VehicleControl(throttle=selected_action[0] * settings.THROT_AM, steer=selected_action[1] * settings.STEER_AM, brake=selected_action[2] * settings.BRAKE_AM))

    return [self.cameras["front"], self.cameras["left"], self.cameras["right"], kmh, action], reward, done
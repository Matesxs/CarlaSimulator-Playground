import time
import carla
import random
from threading import Thread

from helpers.logger import init_logger
import settings

logger = init_logger("Trafic", False)

class TraficControlThread(Thread):
	def __init__(self, client):
		super().__init__()
		self.client = client
		self.daemon = True

		self.traffic_manager = self.client.get_trafficmanager()
		self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

		self.world = self.client.get_world()
		self.spawn_points = self.world.get_map().get_spawn_points()

		self.blueprints = self.world.get_blueprint_library()

		self.vehicle_blueprints = self.blueprints.filter('vehicle.*')
		self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
		self.vehicle_blueprints = [x for x in self.vehicle_blueprints if not x.id.endswith('isetta')]
		self.vehicle_blueprints = [x for x in self.vehicle_blueprints if not x.id.endswith('carlacola')]
		self.vehicle_blueprints = [x for x in self.vehicle_blueprints if not x.id.endswith('cybertruck')]
		self.vehicle_blueprints = [x for x in self.vehicle_blueprints if not x.id.endswith('t2')]

		self.colsenzor_bp = self.blueprints.find("sensor.other.collision")

		self.lights = carla.VehicleLightState.LowBeam
		self.lights |= carla.VehicleLightState.Position

		self.vehicles_list = []

	def spawn_vehicles(self, num_of_vehicles:int):
		if num_of_vehicles == 0: return

		for _ in range(num_of_vehicles):
			blueprint = random.choice(self.vehicle_blueprints)
			if blueprint.has_attribute('color'):
				color = random.choice(blueprint.get_attribute('color').recommended_values)
				blueprint.set_attribute('color', color)
			if blueprint.has_attribute('driver_id'):
				driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
				blueprint.set_attribute('driver_id', driver_id)
			blueprint.set_attribute('role_name', 'autopilot')

			try:
				actor = self.world.spawn_actor(blueprint, random.choice(self.spawn_points))
				if actor:
					actor.set_autopilot(True)
					try:
						actor.set_light_state(carla.VehicleLightState(self.lights))
					except:
						pass

					self.vehicles_list.append(actor)
			except:
				pass

		self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

	def run(self) -> None:
		try:
			self.spawn_vehicles(settings.VEHICLES_TO_KEEP)
			logger.info("Trafic controll started")

			while True:
				vehicles_to_spawn = settings.VEHICLES_TO_KEEP - len(self.vehicles_list)
				if vehicles_to_spawn > 0: self.spawn_vehicles(vehicles_to_spawn)
				time.sleep(settings.TRAFIC_WAIT_LOOP_TIME)
		except RuntimeError:
			pass
		except:
			pass

		logger.info("Trafic controll exited")
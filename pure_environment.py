import time
import carla

import settings
from helpers.environment_control import start_carla, terminate_carla
from modules.weather_control import WeatherControlThread
from modules.trafic_control import TraficControlThread
from helpers.logger import init_logger

logger = init_logger("Pure Environment", False)

def handle_environment():
	client = carla.Client(settings.CONNECTION_IP, settings.CONNECTION_PORT)
	client.set_timeout(20.0)

	logger.info("Starting environment controllers")
	trafic_control = TraficControlThread(client)
	weather_control = WeatherControlThread(client)
	trafic_control.start()
	weather_control.start()
	logger.info("Controllers started")

	try:
		while True:
			client.get_world()
			time.sleep(20)
	except KeyboardInterrupt:
		logger.info("Environment exited")
	except:
		logger.error("Environment failed")
	finally:
		trafic_control.terminate = True
		weather_control.terminate = True

	trafic_control.join()
	weather_control.join()
	logger.info("Environment exited")

if __name__ == '__main__':
	if "localhost" in settings.CONNECTION_IP:
		start_carla(settings.SIM_QUALITY)
	handle_environment()
	if "localhost" in settings.CONNECTION_IP:
		terminate_carla()
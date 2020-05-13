import os
import sys
import time
import numpy as np
import carla

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stdin = sys.stdin
sys.stdin = open(os.devnull, 'w')
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stdin = stdin
sys.stderr = stderr

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except:
		pass

import settings
from helpers.environment_control import start_carla, terminate_carla
from helpers.logger import init_logger
from modules.agent import Agent
from modules.modelhandler import ModelHandler
from modules.weather_control import WeatherControlThread
from modules.trafic_control import TraficControlThread

logger = init_logger("Play", False)

MODEL_WEIGHTS = ""

def play():
	client = carla.Client(settings.CONNECTION_IP, settings.CONNECTION_PORT)
	client.set_timeout(20.0)

	# Create controllers
	trafic_control = TraficControlThread(client)
	weather_control = WeatherControlThread(client)
	trafic_control.start()
	weather_control.start()
	logger.info("Controllers started")

	predicter = ModelHandler(settings.MODEL_NAME, target_weights_path=MODEL_WEIGHTS, train=False)
	agent = Agent(999999, client, False)

	try:
		while True:
			step = 1

			state = agent.spawn()

			while True:
				start_step_time = time.time()

				action = int(np.argmax(predicter.get_qs(state)))
				new_state, _, done = agent.step(action)
				state = new_state

				if done:
					agent.clear_agent()
					break

				time_diff1 = agent.episode_start + step / settings.FPS_COMPENSATION - time.time()
				time_diff2 = start_step_time + 1 / settings.FPS_COMPENSATION - time.time()
				if time_diff1 > 0:
					time.sleep(min(0.125, time_diff1))
				elif time_diff2 > 0:
					time.sleep(min(0.125, time_diff2))
	except KeyboardInterrupt:
		logger.info("Exiting playing - Keyboard interrupt")
	except:
		logger.error("Playing failed")
	finally:
		trafic_control.terminate = True
		weather_control.terminate = True

if __name__ == '__main__':
	if "localhost" in settings.CONNECTION_IP:
		start_carla(settings.SIM_QUALITY)
	play()
	if "localhost" in settings.CONNECTION_IP:
		terminate_carla()
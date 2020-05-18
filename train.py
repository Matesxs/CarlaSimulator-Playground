import random
import os
import sys
import cv2
import time
import math
from threading import Thread
from collections import deque
import json
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
from modules.weather_control import WeatherControlThread
from modules.trafic_control import TraficControlThread
from modules.modelhandler import ModelHandler

logger = init_logger("Training", False)

random.seed(settings.RANDOM_SEED)
tf.random.set_seed(settings.RANDOM_SEED)

def save_models(trainer, model_name, episode, epsilon, score):
	model_name = f"{model_name}__{episode}ep__{epsilon}eps__{score}score.h5"
	path = os.path.join(settings.MODEL_SAVE_PATH, model_name)

	logger.info(f"Saving new record model\n{model_name}")
	trainer.save_weights(path)

class Trainer(Thread):
	def __init__(self, client, identifier, epsilon, get_qs_callbatch, update_replay_memory_callback):
		super().__init__()
		self.daemon = True
		self.client = client

		self.terminate = False
		self.fail_flag = False
		self.halt = False

		self.get_qs = get_qs_callbatch
		self.update_replay_memory = update_replay_memory_callback
		self.identifier = identifier

		self.agent = Agent(identifier, self.client, True)

		self.action = None
		self.episode = 0
		self.epsilon = epsilon
		self.scores_history = deque(maxlen=settings.LOG_EVERY)
		self.score_record = None
		self.steps_per_second = deque(maxlen=settings.LOG_EVERY)

		self.actions_statistic = deque(maxlen=int(settings.LOG_EVERY * settings.SECONDS_PER_EXPISODE * settings.FPS_COMPENSATION))

	def get_action(self, action:int):
		num_of_logged_actions = len(self.actions_statistic)
		if num_of_logged_actions <= 0: return 0
		return self.actions_statistic.count(action) / num_of_logged_actions

	def get_steps_per_second(self):
		if len(self.steps_per_second) > 0:
			return sum(self.steps_per_second) / len(self.steps_per_second)
		return 0

	def get_preview_data(self):
		if self.agent.prev_camera is not None and self.agent.initialized:
			return cv2.cvtColor(self.agent.prev_camera, cv2.COLOR_RGB2BGR)
		return np.zeros((settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[1], settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[0], settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[2]))

	def get_mean_score(self):
		if len(self.scores_history) > 0:
			return sum(self.scores_history) / len(self.scores_history)
		return 0

	def get_episode(self):
		return self.episode

	def run(self) -> None:
		logger.info(f"Trainer {self.identifier} started")

		while not self.terminate:
			if self.halt:
				time.sleep(0.1)
				continue

			reward = None
			episode_reward = 0
			step = 1

			try:
				state = self.agent.spawn()
				self.fail_flag = False
			except:
				self.fail_flag = True
				break

			episode_data_memory = deque()

			while not self.fail_flag:
				start_step_time = time.time()

				if self.epsilon is None or np.random.random() > self.epsilon:
					self.action = int(np.argmax(self.get_qs(state)))
					self.actions_statistic.append(self.action)
				else:
					self.action = random.choice(list(settings.ACTIONS.keys()))

				try:
					new_state, reward, done = self.agent.step(self.action)
				except:
					logger.error(f"Trainer {self.identifier} - Failed to make step")
					self.fail_flag = True
					break

				episode_data_memory.append((state, self.action, reward, new_state, done))
				state = new_state

				episode_reward += reward

				if done:
					self.agent.clear_agent()
					self.action = None
					break

				time_diff1 = self.agent.episode_start + step / settings.FPS_COMPENSATION - time.time()
				time_diff2 = start_step_time + 1 / settings.FPS_COMPENSATION - time.time()
				if time_diff1 > 0:
					time.sleep(min(0.125, time_diff1))
				elif time_diff2 > 0:
					time.sleep(min(0.125, time_diff2))

				step += 1

			if not reward or not self.agent.episode_start: continue

			episode_time = time.time() - self.agent.episode_start
			if episode_time == 0: episode_time = 10 ^ -9
			average_steps_per_second = step / episode_time

			self.steps_per_second.append(average_steps_per_second)

			reward_factor = settings.FPS_COMPENSATION / average_steps_per_second
			episode_reward_weighted = ((episode_reward - reward) * reward_factor + reward) * settings.EPISODE_REWARD_MULTIPLIER

			if episode_time > settings.MINIMUM_EPISODE_LENGTH:
				self.update_replay_memory(episode_data_memory)
				self.scores_history.append(episode_reward_weighted)
				self.episode += 1

			del episode_data_memory

		logger.info(f"Trainer {self.identifier} stopped")

def train(checkpoint=None):
	base_episode = 0
	epsilon = settings.START_EPSILON
	score_record = None
	model_weights = settings.WEIGHT_PATH
	target_model_weights = settings.TARGET_WEIGHTS_PATH

	if checkpoint:
		base_episode = checkpoint["episode"]
		epsilon = checkpoint["epsilon"]
		score_record = checkpoint["score_record"]
		model_weights = checkpoint["weights"]
		target_model_weights = checkpoint["target_weights"]

	last_episode = base_episode
	last_logged_episode = base_episode
	last_checkpoint = base_episode

	def terminate_environment():
		# Stop all agents
		for agent in agents:
			agent.terminate = True

		# Stop trainer
		trainer.terminate = True

		# Stop controllers
		trafic_control.terminate = True
		weather_control.terminate = True

		# Wait for training finish
		trainer.join()

		# Make last checkpoint
		checkpoint_training()

		# Shutdown Carla
		if "localhost" in settings.CONNECTION_IP:
			terminate_carla()

	def get_actions():
		actions = {}
		num_of_agents = len(agents)

		for action in settings.ACTIONS.keys():
			actions[action] = 0
			for agent in agents:
				actions[action] += (agent.get_action(action) / num_of_agents)
		return actions

	def get_steps_per_second():
		sp_p_s = 0
		if agents:
			for agent in agents:
				sp_p_s += agent.get_steps_per_second()
			sp_p_s /= len(agents)
		return round(sp_p_s, 3)

	def get_episode():
		ep = base_episode
		if agents:
			for agent in agents:
				ep += agent.get_episode()
		return ep

	def get_reward():
		reward = 0
		if agents:
			for agent in agents:
				reward += agent.get_mean_score()
			reward /= len(agents)
		return round(reward, 3)

	def get_fail_flags():
		return len([a.identifier for a in agents if a.fail_flag]) > 0

	def set_epsilon():
		for agent in agents:
			agent.epsilon = epsilon

	def checkpoint_training():
		# Create checkpoint paths
		checkpoint_path = os.path.join(settings.CHECKPOINT_SAVE_PATH, trainer.get_model_name())
		if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
		model_path = os.path.join(checkpoint_path, "train_model.h5")
		target_model_path = os.path.join(checkpoint_path, "train_target_model.h5")
		params_path = os.path.join(checkpoint_path, "params.json")

		# Save weights
		trainer.save_weights(model_path)
		trainer.save_target_weights(target_model_path)

		params = {
			"episode": episode,
			"epsilon": epsilon,
			"score_record": score_record,
			"weights": model_path,
			"target_weights": target_model_path
		}

		with open(params_path, "w", encoding='utf-8') as f:
			json.dump(params, f, ensure_ascii=False, indent=4)

	def save_record_model():
		model_path = os.path.join(settings.MODEL_SAVE_PATH, trainer.get_model_name())
		if not os.path.exists(model_path): os.makedirs(model_path)

		model_path = os.path.join(model_path, f"{episode}ep__{epsilon}eps__{reward}rew.h5")
		trainer.save_target_weights(model_path)

	def preview_agents(agents):
		num_of_agents = len(agents)
		num_of_rows = math.floor(num_of_agents / 4)
		if num_of_agents % 4 > 0: num_of_rows += 1

		prev_image = np.zeros((settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[1] * num_of_rows, settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[0] * 4, settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[2]), np.uint8)
		for agent in agents:
			idx = agent.identifier

			row = math.floor(idx / 4)
			col = idx - (row * 4)

			y1 = row * settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[1]
			y2 = y1 + settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[1]
			x1 = col * settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[0]
			x2 = x1 + settings.PREVIEW_CAMERA_IMAGE_DIMENSIONS[0]

			agent_img = agent.get_preview_data()
			agent_img = cv2.putText(agent_img, f'Agent {agent.identifier}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 30, 0), 2, cv2.LINE_AA)
			agent_img = cv2.putText(agent_img, f'A: {agent.action}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 30, 0), 2, cv2.LINE_AA)
			prev_image[y1 : y2, x1 : x2] = agent_img

		if settings.EPISODES is not None:
			prev_image = cv2.putText(prev_image, f"Ep: {episode}/{settings.EPISODES} - Steps per second (Crude FPS): {steps_per_second}, Reward: {reward}, Epsilon: {round(epsilon, 5)}", (10, prev_image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2, cv2.LINE_AA)
		else:
			prev_image = cv2.putText(prev_image, f"Ep: {episode} - Steps per second (Crude FPS): {steps_per_second}, Reward: {reward}, Epsilon: {round(epsilon, 5)}", (10, prev_image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
		prev_image = cv2.resize(prev_image, dsize=None, fx=0.5, fy=0.5)

		cv2.imshow("Preview", prev_image)
		cv2.waitKey(1)

	assert settings.NUM_OF_AGENTS > 0, "Invalid number of agents"
	if settings.EPISODES:
		assert base_episode < settings.EPISODES, "Training already finished"

	logger.info("Initializing environment")
	# Start Carla and connect to it
	if "localhost" in settings.CONNECTION_IP:
		start_carla(settings.SIM_QUALITY)

	client = carla.Client(settings.CONNECTION_IP, settings.CONNECTION_PORT)
	client.set_timeout(20.0)

	# Create controllers
	trafic_control = TraficControlThread(client)
	weather_control = WeatherControlThread(client)
	trafic_control.start()
	weather_control.start()
	logger.info("Controllers started")

	# Start trainer
	trainer = ModelHandler(settings.MODEL_NAME, weights_path=model_weights, target_weights_path=target_model_weights, train=True)
	trainer.tensorboard.step = base_episode
	trainer.__last_logged_episode = base_episode
	trainer.start()
	logger.info("Trainer started")

	# Create agents
	agents = [Trainer(client, i, epsilon, trainer.get_qs, trainer.update_replay_memory) for i in range(settings.NUM_OF_AGENTS)]
	logger.info("Agents created")
	time.sleep(2)

	# Start agents
	for agent in agents:
		agent.start()
		time.sleep(0.05)
	logger.info("Environment started")

	halted_training = False

	try:
		episode = get_episode()
		reward = get_reward()
		steps_per_second = get_steps_per_second()

		while True:
			episode = get_episode()

			# If target episode is reached, exit training loop
			if settings.EPISODES is not None:
				if episode >= settings.EPISODES:
					logger.info("Training finished")
					break

			# Perform only on unique episode
			if last_episode != episode:
				reward = get_reward()
				steps_per_second = get_steps_per_second()

				# Log stats every 10 episodes
				if (last_logged_episode + settings.LOG_EVERY) <= episode:
					if settings.EPISODES is not None:
						logger.info(f"Ep: {episode}/{settings.EPISODES} - Steps per second (Crude FPS): {steps_per_second}, Reward: {reward}, Epsilon: {epsilon}")
					else:
						logger.info(f"Ep: {episode} - Steps per second (Crude FPS): {steps_per_second}, Reward: {reward}, Epsilon: {epsilon}")

					# Pause training when steps per settings are too low
					if steps_per_second < settings.MIN_FPS_FOR_TRAINING and not halted_training:
						trainer.switch_halt_state()
						halted_training = True
					elif steps_per_second > settings.MIN_FPS_FOR_TRAINING and halted_training:
						trainer.switch_halt_state()
						halted_training = False

					last_logged_episode = (episode // settings.LOG_EVERY) * settings.LOG_EVERY

				# Decay epsilon and update it on all agents
				if epsilon > settings.MIN_EPSILON and not halted_training:
					epsilon *= settings.EPSILON_DECAY
					epsilon = max(settings.MIN_EPSILON, epsilon)
					set_epsilon()

				if (base_episode + (len(agents) * 2)) < episode:
					# Checkpoint training
					if (last_checkpoint + settings.CHECKPOINT_EVERY) <= episode:
						# Log stats to tensorboard

						try:
							trainer.tensorboard.update_stats(step=episode, steps_per_second=steps_per_second, reward=reward, epsilon=epsilon, actions=get_actions())
							trainer.tensorboard.step = episode
						except Exception as e:
							logger.warning(f"Failed to write tensorboard data\n```{e}```")

						checkpoint_training()
						last_checkpoint = (episode // settings.CHECKPOINT_EVERY) * settings.CHECKPOINT_EVERY

					if score_record:
						if reward >= settings.MIN_SCORE_TO_SAVE and epsilon <= settings.MAX_EPSILON_TO_SAVE and reward > (score_record + settings.MIN_SCORE_DIF):
							save_record_model()
							score_record = reward
					else:
						if reward >= settings.MIN_SCORE_TO_SAVE and epsilon <= settings.MAX_EPSILON_TO_SAVE:
							save_record_model()
							score_record = reward

				last_episode = episode

			if settings.PREVIEW_AGENTS:
				preview_agents(agents)

			# Stop on fail
			if get_fail_flags():
				logger.info("Stopping environment")
				break
				
			time.sleep(0.001)
	except KeyboardInterrupt:
		logger.info("Exiting - Keyboard Interrupt")
	except Exception as e:
		logger.exception(e)
	finally:
		# Destroy preview window and close environment
		cv2.destroyAllWindows()
		terminate_environment()

if __name__ == '__main__':
	model_name = f"{settings.MODEL_NAME}"
	if settings.FEED_SPEED_INPUT:
		model_name += "__SPEED"
	if settings.FEED_LAST_ACTION_INPUT:
		model_name += "__lst_ACT"

	checkpoint_path = os.path.join(settings.CHECKPOINT_SAVE_PATH, model_name)
	params_path = os.path.join(checkpoint_path, "params.json")

	if os.path.exists(params_path):
		logger.info("Loading checkpoint")

		f = open(params_path, "rb")
		params = json.load(f)
		f.close()

		if params:
			train(params)
	else:
		train()
import sys
from collections import deque
from typing import Union
import random
import time
import os
import numpy as np
from threading import Thread
from modules.custom_tensorboard import TensorBoardCustom
from tensorflow.keras.utils import plot_model
import gc

from modules.models import create_model
from helpers.logger import init_logger
import settings

logger = init_logger("ModelHandler", False)

action_normalizer_value = float(len(settings.ACTIONS) - 1)

class ModelHandler(Thread):
  def __init__(self, model:str, weights_path:str=None, target_weights_path:str=None, train:bool=True):
    super().__init__()
    self.daemon = True
    self.terminate = False
    self.__training = train
    self.__halt_training = False

    self.__last_logged_episode = None

    self.__model = create_model(model, target_weights_path, False)
    self.__train_model = None
    self.__target_model = None

    logger.info("Model created")
    if not self.__model:
      logger.error("Unable to create models")
      sys.exit(1)

    if train:
      self.__train_model = create_model(model, weights_path, train)
      self.__target_model = create_model(model, target_weights_path, False)
      logger.info("Training models created")

      if not self.__target_model or not self.__train_model:
        logger.error("Unable to create training models")
        sys.exit(1)

      if weights_path:
        if not target_weights_path:
          self.__override_weights()
          logger.info("Weights synchronized")

      self.__replay_memory = deque(maxlen=settings.REPLAY_MEMORY_SIZE)
      self.__target_update_counter = 0

      if not os.path.exists(f"logs/{self.__model.name}"): os.makedirs(f"logs/{self.__model.name}")
      self.tensorboard = TensorBoardCustom(log_dir=f"logs/{self.__model.name}")

    plot_model(self.__model, f"{self.__model.name}.png", show_shapes=True, expand_nested=True)
    logger.info("Trainer initialized")

  def print_model_summary(self):
    if self.__model:
      self.__model.summary()

  def switch_halt_state(self):
    if not self.__training: return None
    self.__halt_training = not self.__halt_training
    if self.__halt_training:
      logger.info("Training halted")
    else:
      logger.info("Training resumed")
    return self.__halt_training

  def get_model_name(self):
    return self.__model.name

  def update_replay_memory(self, transition:Union[deque, list]):
    if self.__training:
      self.__replay_memory += transition
    else:
      logger.warning("Cant append data when training flag is not set")

  def save_weights(self, path):
    if self.__train_model:
      self.__train_model.save_weights(path)

  def save_target_weights(self, path):
    if self.__target_model:
      self.__target_model.save_weights(path)

  def __override_weights(self):
    if self.__train_model and self.__target_model:
      self.__target_model.set_weights(self.__train_model.get_weights())
      self.__model.set_weights(self.__train_model.get_weights())

  def get_qs(self, state:Union[np.ndarray, list]):
    Xs = [state[0].reshape(-1, *state[0].shape, 1) / 255.0, state[1].reshape(-1, *state[1].shape, 1) / 255.0, state[2].reshape(-1, *state[2].shape, 1) / 255.0]

    if settings.FEED_SPEED_INPUT:
      Xs.append((np.array([state[3]]) - 50.0) / 50.0)
    if settings.FEED_LAST_ACTION_INPUT:
      Xs.append(np.array([state[4]]) / action_normalizer_value)

    prediction = self.__model.predict(Xs)
    return prediction[0]

  def __train(self):
    if len(self.__replay_memory) < settings.MIN_REPLAY_MEMORY_SIZE: return

    minibatch = random.sample(self.__replay_memory, settings.MINIBATCH_SIZE)

    current_states = [np.array([np.expand_dims(transition[0][0], 2) for transition in minibatch]) / 255.0,
                      np.array([np.expand_dims(transition[0][1], 2) for transition in minibatch]) / 255.0,
                      np.array([np.expand_dims(transition[0][2], 2) for transition in minibatch]) / 255.0]

    if settings.FEED_SPEED_INPUT:
      current_states.append((np.array([transition[0][3] for transition in minibatch]) - 50.0) / 50.0)
    if settings.FEED_LAST_ACTION_INPUT:
      current_states.append(np.array([transition[0][4] for transition in minibatch]) / action_normalizer_value)

    current_qs_list = self.__train_model.predict(current_states, settings.PREDICTION_BATCH_SIZE)


    new_current_states = [np.array([np.expand_dims(transition[3][0], 2) for transition in minibatch]) / 255.0,
                          np.array([np.expand_dims(transition[3][1], 2) for transition in minibatch]) / 255.0,
                          np.array([np.expand_dims(transition[3][2], 2) for transition in minibatch]) / 255.0]

    if settings.FEED_SPEED_INPUT:
      new_current_states.append((np.array([transition[3][3] for transition in minibatch]) - 50.0) / 50.0)
    if settings.FEED_LAST_ACTION_INPUT:
      new_current_states.append(np.array([transition[3][4] for transition in minibatch]) / action_normalizer_value)

    future_qs_list = self.__target_model.predict(new_current_states, settings.PREDICTION_BATCH_SIZE)

    X_front = []
    X_left = []
    X_right = []
    X_speed = []
    X_lst_action = []

    y = []

    for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
      if not done:
        max_future_q = np.max(future_qs_list[index])
        new_q = reward + settings.DISCOUNT * max_future_q
      else:
        new_q = reward

      current_qs = current_qs_list[index]
      current_qs[action] = new_q

      X_front.append(np.expand_dims(current_state[0], 2))
      X_left.append(np.expand_dims(current_state[1], 2))
      X_right.append(np.expand_dims(current_state[2], 2))

      if settings.FEED_SPEED_INPUT:
        X_speed.append(current_state[3])
      if settings.FEED_LAST_ACTION_INPUT:
        X_lst_action.append(current_state[4])

      y.append(current_qs)

    Xs = [np.array(X_front) / 255, np.array(X_left) / 255, np.array(X_right) / 255]
    if settings.FEED_SPEED_INPUT:
      Xs.append((np.array(X_speed) - 50) / 50)
    if settings.FEED_LAST_ACTION_INPUT:
      Xs.append(np.array(X_lst_action) / action_normalizer_value)

    log_it = False
    if self.__last_logged_episode != self.tensorboard.step:
      log_it = True
      self.__last_logged_episode = self.tensorboard.step
      self.tensorboard.log_weights(self.__train_model)

    self.__train_model.fit(Xs, np.array(y), batch_size=settings.TRAINING_BATCH_SIZE, verbose=0, shuffle=False, epochs=settings.TRAINING_EPOCHS, callbacks=[self.tensorboard] if log_it else None)

    del Xs
    del X_front
    del X_left
    del X_right
    del X_speed
    del X_lst_action
    del current_states
    del new_current_states
    del future_qs_list
    del current_qs_list

    self.__target_update_counter += 1

  def run(self) -> None:
    if not self.__training:
      logger.warning("Training flag not set!")
      return

    damn_data = [np.zeros((1, settings.CAMERA_IMAGE_DIMENSIONS[1], settings.CAMERA_IMAGE_DIMENSIONS[0], settings.CAMERA_IMAGE_DIMENSIONS[2])),
                 np.zeros((1, settings.CAMERA_IMAGE_DIMENSIONS[1], settings.CAMERA_IMAGE_DIMENSIONS[0], settings.CAMERA_IMAGE_DIMENSIONS[2])),
                 np.zeros((1, settings.CAMERA_IMAGE_DIMENSIONS[1], settings.CAMERA_IMAGE_DIMENSIONS[0], settings.CAMERA_IMAGE_DIMENSIONS[2]))]
    if settings.FEED_SPEED_INPUT:
      damn_data.append(np.array([0]))
    if settings.FEED_LAST_ACTION_INPUT:
      damn_data.append(np.array([0]))

    self.__model.predict(damn_data)
    del damn_data

    while len(self.__replay_memory) < settings.MIN_REPLAY_MEMORY_SIZE and not self.terminate:
      time.sleep(0.1)

    logger.info("Training started")
    while not self.terminate:
      while self.__halt_training and not self.terminate:
        time.sleep(0.1)

      self.__train()

      if self.__target_update_counter > settings.UPDATE_TARGET_EVERY:
        self.__override_weights()
        self.__target_update_counter = 0
        gc.collect()

      time.sleep(0.05)

    logger.info("Training stopped")
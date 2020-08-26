import carla
import time
import atexit
import sys
import os
import subprocess
import psutil

import settings
from helpers.logger import init_logger

logger = init_logger("ENV Controll", False)

def operating_system():
  logger.debug(os.name)
  return 'windows' if os.name == 'nt' else 'linux'

def get_binary():
  return 'CarlaUE4.exe' if operating_system() == 'windows' else 'CarlaUE4.sh'

def get_exec_command():
  binary = get_binary()
  exec_command = binary if operating_system() == 'windows' else ('./' + binary)

  return binary, exec_command

def terminate_process(binary):
  for process in psutil.process_iter():
    if process.name().lower().startswith(binary.split('.')[0].lower()) or process.name().lower().startswith(binary.lower()):
      try:
        process.terminate()
      except:
        pass

  still_alive = []
  for process in psutil.process_iter():
    if process.name().lower().startswith(binary.split('.')[0].lower()) or process.name().lower().startswith(binary.lower()):
      still_alive.append(process)

  if len(still_alive):
    for process in still_alive:
      try:
        process.kill()
      except:
        pass
    psutil.wait_procs(still_alive)

def terminate_carla():
  binary = get_binary()
  terminate_process(binary)

# noinspection PyArgumentList
def start_carla(sim_quality:str="Epic"):
  atexit.register(terminate_carla)

  logger.info('Starting Carla...')
  terminate_carla()

  if not os.path.isdir(settings.PATH_TO_MAIN_CARLA_FOLDER):
    logger.error("Carla not found!")
    sys.exit(-1)

  logger.debug(get_exec_command()[1] + f" -carla-rpc-port=2000 -carla-server -quality-level={sim_quality} -fps={settings.FPS_COMPENSATION}")
  subprocess.Popen(get_exec_command()[1] + f" -carla-rpc-port=2000 -carla-server -quality-level={sim_quality} -fps={settings.FPS_COMPENSATION}", cwd=settings.PATH_TO_MAIN_CARLA_FOLDER, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

  while True:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    try:
      client.get_world()
      logger.info("Carla started...")
      break
    except:
      time.sleep(0.1)

def restart_carla(sim_quality:str="Epic"):
  logger.info("Restarting Carla")
  terminate_carla()
  time.sleep(10)
  start_carla(sim_quality)
  time.sleep(5)
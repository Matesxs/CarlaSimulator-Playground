import subprocess
import time
import settings
from helpers.logger import init_logger

logger = init_logger("Infinite trainer", False)

num_of_restarts = 0

if __name__ == '__main__':
  tbp = None
  if settings.START_TENSORBOARD_ON_TRAINING:
    tbp = subprocess.Popen(f"./venv/Scripts/python.exe -m tensorboard.main --logdir logs", stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  p = subprocess.Popen("./venv/Scripts/python.exe train.py")
  while True:
    try:
      p.wait()
      logger.info("Restarting environment in 5s")
      time.sleep(5)
      p = subprocess.Popen("./venv/Scripts/python.exe train.py")
      num_of_restarts += 1
    except KeyboardInterrupt:
      p.send_signal(subprocess.signal.CTRL_C_EVENT)
      logger.info("Training interrupted")
      break
    except Exception as e:
      logger.error(f"Infinite loop exception\n{e}")

  if tbp:
    tbp.send_signal(subprocess.signal.SIGKILL)

  logger.info("Infinite trainer finished")
  logger.info(f"Restarts count: {num_of_restarts}")
## Simulation
PATH_TO_MAIN_CARLA_FOLDER = "lib/Carla"

CONNECTION_IP = "localhost"
CONNECTION_PORT = 2000

SIM_QUALITY = "Low" # Low / Epic

NUM_OF_AGENTS = 8
PREVIEW_AGENTS = True

FPS_COMPENSATION = 20.0
LOG_EVERY = 10

# End thresholds
ROTATION_THRESHOLD = 75
COLLISION_FILTER = [['static.sidewalk', -1], ['static.road', -1], ['vehicle.', 100]]

SECONDS_PER_EXPISODE = 30.0

# Valid episode criteria
MINIMUM_EPISODE_LENGTH = 4.0
MINIMUM_STEPS = 50

# Reward Settings
EPISODE_REWARD_MULTIPLIER = 100
BAD_ROTATION_REWARD = -1.0
COLLISION_REWARD = -1.0
SPEED_MIN_REWARD = -0.007
SPEED_MAX_REWARD = 0.005
TIME_WEIGHTED_REWARDS = False

# Trafic settings
VEHICLES_TO_KEEP = 30
TRAFIC_WAIT_LOOP_TIME = 10
##end

## Agent settings
STEER_AM = 1.0
THROT_AM = 1.0
BRAKE_AM = 0.8

# Camera settings
PREVIEW_CAMERA_IMAGE_DIMENSIONS = (540, 480, 3)
PREVIEW_CAMERA_FOV = 90
CAMERA_IMAGE_DIMENSIONS = (480, 270, 1)
CAMERA_FOV = 130
##end

## Training settings
MIN_FPS_FOR_TRAINING = 7

# Model save settings
MIN_SCORE_TO_SAVE = -200
MIN_SCORE_DIF = 10
MAX_EPSILON_TO_SAVE = 0.3
CHECKPOINT_EVERY = 50

MODEL_SAVE_PATH = "models"
CHECKPOINT_SAVE_PATH = "checkpoint"

DEFAULT_ACTION = 7
ACTIONS = {
	0: [0.7, -1.0, 0.0],
	1: [1.0, 0.0, 0.0],
	2: [0.7, 1.0, 0.0],
	3: [0.0, -1.0, 0.0],
	4: [0.0, 0.0, 0.0],
	5: [0.0, 1.0, 0.0],
	6: [0.0, -1.0, 1.0],
	7: [0.0, 0.0, 1.0],
	8: [0.0, 1.0, 1.0]
}

# Num of episodes to train for / None - infinite
EPISODES = None

UPDATE_TARGET_EVERY = 100
TRAINING_EPOCHS = 1

# Batches
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4

# Model settings
MODEL_NAME = "CNN_128_3x64"
WEIGHT_PATH = None
TARGET_WEIGHTS_PATH = None

FEED_LAST_ACTION_INPUT = True
FEED_SPEED_INPUT = True
HIDDEN_LAYERS = {
	"hidden_dense_1": 256
}

# Optimizer settings
OPTIMIZER = "adam" # sgd, adam, adadelta
OPTIMIZER_LEARNING_RATE = 0.001
OPTIMIZER_DECAY = 0
SGD_MOMENTUM = 0.2

# DQN Settings
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 20_000
MIN_REPLAY_MEMORY_SIZE = 5_000

# Random rate settings
START_EPSILON = 0.9
EPSILON_DECAY = 0.99997 # 0.99995, 0.99985
MIN_EPSILON = 0.1
##end

# Mic
RANDOM_SEED = 12586
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, AveragePooling2D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.initializers import RandomNormal

import settings
from helpers.logger import init_logger

logger = init_logger("Models", False)

kernel_initializer = RandomNormal(stddev=0.02)

image_shape = (settings.CAMERA_IMAGE_DIMENSIONS[1], settings.CAMERA_IMAGE_DIMENSIONS[0], settings.CAMERA_IMAGE_DIMENSIONS[2])
front_camera_input = Input(name="front_camera_input", shape=image_shape)
right_camera_input = Input(name="right_camera_input", shape=image_shape)
left_camera_input = Input(name="left_camera_input", shape=image_shape)
# noinspection PyTypeChecker
combined_cameras_input = Concatenate(name="combine_cameras_concanate")([front_camera_input, left_camera_input, right_camera_input])

def CNN_128_3x64():
	combined_camera = Conv2D(128, (3, 3), padding='same', name="combine_camera_conv2d_1", activation="relu", kernel_initializer=kernel_initializer)(combined_cameras_input)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_1")(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', name="combine_camera_conv2d_2", activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_2")(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", name="combine_camera_conv2d_3", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_3")(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", name="combine_camera_conv2d_4", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_4")(combined_camera)

	combined_camera = Flatten(name="combine_camera_flatten")(combined_camera)

	return combined_camera

def CNN_4x64():
	combined_camera = Conv2D(64, (3, 3), padding='same', name="combine_camera_conv2d_1", activation="relu", kernel_initializer=kernel_initializer)(combined_cameras_input)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_1")(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', name="combine_camera_conv2d_2", activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_2")(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", name="combine_camera_conv2d_3", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_3")(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", name="combine_camera_conv2d_4", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_4")(combined_camera)

	combined_camera = Flatten(name="combine_camera_flatten")(combined_camera)

	return combined_camera

def CNN_base_4():
	combined_camera = Conv2D(256, (3, 3), padding='same', name="combine_camera_conv2d_1", activation="relu", kernel_initializer=kernel_initializer)(combined_cameras_input)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_1")(combined_camera)

	combined_camera = Conv2D(128, (3, 3), padding='same', name="combine_camera_conv2d_2", activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_2")(combined_camera)

	combined_camera = Conv2D(128, (3, 3), padding='same', activation="relu", name="combine_camera_conv2d_3", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_3")(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", name="combine_camera_conv2d_4", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same', name="combine_camera_avgpooling2d_4")(combined_camera)

	combined_camera = Flatten(name="combine_camera_flatten")(combined_camera)

	return combined_camera

def create_model(model_name:str, weights_path:str=None, compile:bool=True) -> Model:
	try:
		output = globals()[model_name]()
	except:
		logger.error("Unknown model")
		sys.exit(1)

	inputs = [front_camera_input, left_camera_input, right_camera_input]
	parts_to_concanate = [output]

	if settings.FEED_SPEED_INPUT:
		speed_input = Input(name="speed_input", shape=(1,))
		inputs.append(speed_input)
		parts_to_concanate.append(speed_input)

	if settings.FEED_LAST_ACTION_INPUT:
		lst_action_input = Input(name="last_action_input", shape=(1,))
		inputs.append(lst_action_input)
		parts_to_concanate.append(lst_action_input)

	if len(parts_to_concanate) > 1:
		# noinspection PyTypeChecker
		x = Concatenate(name="hidden_concanate")(parts_to_concanate)
	else:
		x = output

	for layer_name in settings.HIDDEN_LAYERS.keys():
		x = Dense(settings.HIDDEN_LAYERS[layer_name], activation="linear", name=layer_name)(x)
	predictions = Dense(len(settings.ACTIONS.keys()), activation='linear', name="predictions")(x)

	model_name = f"{settings.MODEL_NAME}"
	if settings.FEED_SPEED_INPUT:
		model_name += "__SPEED"
	if settings.FEED_LAST_ACTION_INPUT:
		model_name += "__LST_ACT"

	model = Model(inputs=inputs, outputs=predictions, name=model_name)
	
	if compile:
		if settings.OPTIMIZER == "adam":
			model.compile(loss="mse", optimizer=Adam(lr=settings.OPTIMIZER_LEARNING_RATE, decay=settings.OPTIMIZER_DECAY), metrics=["accuracy"])
		elif settings.OPTIMIZER == "sgd":
			model.compile(loss="mse", optimizer=SGD(lr=settings.OPTIMIZER_LEARNING_RATE, momentum=settings.SGD_MOMENTUM, decay=settings.OPTIMIZER_DECAY), metrics=["accuracy"])
		elif settings.OPTIMIZER == "adadelta":
			model.compile(loss="mse", optimizer=Adadelta(), metrics=["accuracy"])
		else:
			logger.error("Unknown optimizer for compilation")
			sys.exit(1)

	if weights_path:
		try:
			model.load_weights(weights_path)
		except:
			logger.error("Unable to load weights")
			sys.exit(1)

	return model
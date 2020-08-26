import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dropout, Activation
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

# noinspection PyTypeChecker
def CNN_5_residual():
	cnn_1 = Conv2D(64, (7, 7), padding='same', kernel_initializer=kernel_initializer)(combined_cameras_input)
	cnn_1a = Activation('relu')(cnn_1)
	cnn_1c = Concatenate()([cnn_1a, combined_cameras_input])
	cnn_1ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_1c)

	cnn_2 = Conv2D(64, (5, 5), padding='same', kernel_initializer=kernel_initializer)(cnn_1ap)
	cnn_2a = Activation('relu')(cnn_2)
	cnn_2c = Concatenate()([cnn_2a, cnn_1ap])
	cnn_2ap = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(cnn_2c)

	cnn_3 = Conv2D(128, (5, 5), padding='same', kernel_initializer=kernel_initializer)(cnn_2ap)
	cnn_3a = Activation('relu')(cnn_3)
	cnn_3c = Concatenate()([cnn_3a, cnn_2ap])
	cnn_3ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_3c)

	cnn_4 = Conv2D(256, (5, 5), padding='same', kernel_initializer=kernel_initializer)(cnn_3ap)
	cnn_4a = Activation('relu')(cnn_4)
	cnn_4c = Concatenate()([cnn_4a, cnn_3ap])
	cnn_4ap = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(cnn_4c)

	cnn_5 = Conv2D(512, (3, 3), padding='same', kernel_initializer=kernel_initializer)(cnn_4ap)
	cnn_5a = Activation('relu')(cnn_5)
	cnn_5ap = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(cnn_5a)

	flatten = Flatten()(cnn_5ap)

	return flatten

def CNN_4_base():
	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_cameras_input)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Flatten(name="combine_camera_flatten")(combined_camera)

	return combined_camera

def CNN_3_base():
	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_cameras_input)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Flatten(name="combine_camera_flatten")(combined_camera)

	return combined_camera

def CNN_4_h2l():
	combined_camera = Conv2D(256, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_cameras_input)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Conv2D(128, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Conv2D(128, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

	combined_camera = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=kernel_initializer)(combined_camera)
	combined_camera = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(combined_camera)

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
		if "dense" in layer_name:
			x = Dense(settings.HIDDEN_LAYERS[layer_name], activation="relu", name=layer_name)(x)
		elif "dropout" in layer_name:
			x = Dropout(settings.HIDDEN_LAYERS[layer_name], name=layer_name)(x)
		else:
			logger.warning(f"Invalid layer name: {layer_name}")
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
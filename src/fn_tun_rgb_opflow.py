import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/home/elopez/har_test/source')
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from data_processing.prepare_data import load_dataset_file, load_video, label_processor, get_data

from sklearn.metrics import confusion_matrix, classification_report
from tcn import TCN, tcn_full_summary

from model.backbones import build_feature_extractor, MODEL_ID


IMG_SIZE = 224
MAX_SEQ_LENGTH = 30
NUM_FEATURES = 1536
BATCH_SIZE = 1024
class_vocab = 4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule)


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

log_filename = "log_11_mar_op_flow_convnext.txt"#'log_07_full_mar_t01.txt'
save_logs = True

def write_logs(log):
    try:
        file = open(log_filename, 'x') if not os.path.exists(log_filename) else open(log_filename, 'a')
        file.write(log)
    except e:
        print("Error during logs writing \n", e)
    finally:
        file.close() 

def get_reports(model, chck_file, data, labels):
    
    if os.path.exists(chck_file):
        model.load_weights(chck_file)

    y_pred = model.predict(data, verbose=0)
    predicted_categories = tf.argmax(y_pred, axis=1)
    
    report = classification_report(labels, predicted_categories)
    if save_logs:
        write_logs(report)
    print(report)
    
def prepare_videos(x, feature_extractor, max_seq_length=30, num_features=2048, img_shape=(224, 224), min_frames=7):
    ''' 
        
    '''
    num_samples = len(x)
    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    #frame_masks = np.zeros(shape=(num_samples, max_seq_length), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, max_seq_length, num_features), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(x):
        # Gather all its frames and add a batch dimension.
        #print(str(path))
        if os.path.exists:
            frames = load_video(path.decode('utf-8'), resize=img_shape)
            
            if len(frames) > min_frames:
                frames = frames[None, ...]
                temp_frame_features = np.zeros(
                    shape=(1, max_seq_length, num_features), dtype="float32"
                )

                # Extract features from the frames of the current video.
                for i, batch in enumerate(frames):
                    video_length = batch.shape[0]
                    length = min(max_seq_length, video_length)
                    for j in range(length):
                        temp_frame_features[i, j, :] = feature_extractor.predict(
                            batch[None, j, :],
                            verbose=0
                        )

                frame_features[idx,] = temp_frame_features#.squeeze()
        else:
            print(f'File {path} not found')

    return frame_features

def get_dataset(filepath, split=0, ds_file='info.csv', batch_size=24, max_seq_length=30, debug=False, img_shape=(224,224)):
    df = load_dataset_file(os.path.join(filepath, ds_file), debug=debug)
    
    tmp_df = df[df['split']==split]    
    tmp_df['video_path'] = [os.path.join(filepath, tmp_df.classname.loc[idx], tmp_df.filename.loc[idx]) for idx in tmp_df.index.tolist()]
    #data, labels = prepare_all_videos(tmp_df, max_seq_length=max_seq_length, img_shape=img_shape)
    labels = label_processor(tmp_df)
    
    dataset = tf.data.Dataset.from_tensor_slices((tmp_df['video_path'], labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return dataset 

@tf.function
def train_step(x, y, model):
    with tf.GradientTape() as tape:
        print(x.shape)
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        
        loss_value += sum(model.losses)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y, model):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

@tf.keras.saving.register_keras_serializable()
class ActivityRegularizationLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs

def train(model, feature_extractor, train_ds, test_ds, batch_size=24, epochs = 2, num_features=2048):
    
    for epoch in range(epochs):
        
        print("\nStart of epoch %d" % (epoch,))
        if save_logs:
            write_logs("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                #print(x_batch_train)
                x_data = prepare_videos(x_batch_train.numpy(), feature_extractor, num_features=num_features)
                #print(x_data.shape)
                
                loss_value = train_step(x_data, y_batch_train, model)


            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                    
                )
                if save_logs:
                    write_logs("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                print("Seen so far: %s samples" % ((step + 1) * batch_size))            
            
        train_acc = train_acc_metric.result()
        
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        if save_logs:
            write_logs("Training acc over epoch: %.4f \n" % (float(train_acc),))
        train_acc_metric.reset_states()
        
        for x_batch_test, y_batch_test in test_ds:
            x_test = prepare_videos(x_batch_test.numpy(), feature_extractor, num_features=num_features)
            test_step(x_test, y_batch_test, model)
            
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        
        if save_logs:
            write_logs("Validation acc: %.4f \n" % (float(val_acc),))
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        if save_logs:
            write_logs("Time taken: %.2fs \n" % (time.time() - start_time))
    return model

def get_model_tcn(max_seq_length=30, num_features=2048):

    frame_features_input = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm_Layer_1')(tf.keras.Input((max_seq_length, num_features)))
    
    #x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm1')(frame_features_input)
    x = TCN(input_shape=(80, num_features), nb_filters=64, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32], name='TCN_layer_01')(frame_features_input)
    x = tf.keras.layers.MaxPool1D(3, name='Stream1_MaxPool_layer_01')(x)
    x = TCN(input_shape=(80, num_features), nb_filters=128, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32], name='TCN_layer_02')(x)
    x = tf.keras.layers.MaxPool1D(3, name='Stream1_MaxPool_layer_02')(x)
    #x = TCN(input_shape=(80, num_features), nb_filters=128, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32])(x)
    #x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='Stream1_BatchNorm_Layer_1')(x)  
    x = TCN(input_shape=(80, num_features), nb_filters=64, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32], name='TCN_layer_03')(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name='Stream1_GlobalPool_layer_01')(x)
        
  
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm_Layer_2')(x)
    x = tf.keras.layers.Dropout(0.6, name='dropout')(x)
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    #xx = keras.layers.Dense(len(class_vocab), activation="softmax")(xx)
    output = tf.keras.layers.Dense(class_vocab, activation="softmax")(x)
    rnn_model = tf.keras.Model(frame_features_input, output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

def get_model_lstm(max_seq_length=30, num_features=2048):

    frame_features_input = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm_Layer_1')(tf.keras.Input((max_seq_length, num_features)))
    
    #y = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001)(frame_features_input)  
    y = tf.keras.layers.LSTM(256, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.0001), name='LSTM_layer_01')(frame_features_input)
    y = tf.keras.layers.Conv1D(64, 3, input_shape=(None,max_seq_length, num_features ),padding='same', name='CNN_layer_01')(y)
    y = tf.keras.layers.MaxPool1D(3, name='Stream2_MaxPool_layer_01')(y)
    y = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='Stream2_BatchNorm_Layer_1')(y)  
    y = tf.keras.layers.LSTM(128, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.0001), name='LSTM_layer_02')(y)
    y = tf.keras.layers.Conv1D(128, 3, input_shape=(None,max_seq_length, num_features ),padding='same', name='CNN_layer_02')(y)   
    y = tf.keras.layers.MaxPool1D(3, name='Stream2_MaxPool_layer_02')(y)
    y = tf.keras.layers.LSTM(64, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.0001), name='LSTM_layer_03')(y)
    
    #agregar rnns checar
    
    y = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm_Layer_2')(y)
    y = tf.keras.layers.Dropout(0.6, name='dropout')(y)
    y = tf.keras.layers.Flatten()(y)
    
    y = tf.keras.layers.Dense(8, activation="relu")(y)
    #xx = keras.layers.Dense(len(class_vocab), activation="softmax")(xx)
    output = tf.keras.layers.Dense(class_vocab, activation="softmax")(y)
    rnn_model = tf.keras.Model(frame_features_input, output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

def get_model_gru(max_seq_length=30, num_features=2048):

    frame_features_input = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm_Layer_1')(tf.keras.Input((max_seq_length, num_features)))
    
    #y = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001)(frame_features_input)  
    y = tf.keras.layers.GRU(256, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.0001), name='LSTM_layer_01')(frame_features_input)
    y = tf.keras.layers.Conv1D(64, 3, input_shape=(None,max_seq_length, num_features ),padding='same', name='CNN_layer_01')(y)
    y = tf.keras.layers.MaxPool1D(3, name='Stream2_MaxPool_layer_01')(y)
    y = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='Stream2_BatchNorm_Layer_1')(y)  
    y = tf.keras.layers.GRU(128, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.0001), name='LSTM_layer_02')(y)
    y = tf.keras.layers.Conv1D(128, 3, input_shape=(None,max_seq_length, num_features ),padding='same', name='CNN_layer_02')(y)   
    y = tf.keras.layers.MaxPool1D(3, name='Stream2_MaxPool_layer_02')(y)
    y = tf.keras.layers.GRU(64, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.0001), name='LSTM_layer_03')(y)
    
    #agregar rnns checar
    
    y = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm_Layer_2')(y)
    y = tf.keras.layers.Dropout(0.6, name='dropout')(y)
    y = tf.keras.layers.Flatten()(y)
    
    y = tf.keras.layers.Dense(8, activation="relu")(y)
    #xx = keras.layers.Dense(len(class_vocab), activation="softmax")(xx)
    output = tf.keras.layers.Dense(class_vocab, activation="softmax")(y)
    rnn_model = tf.keras.Model(frame_features_input, output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


def get_model(max_seq_length=30, num_features=2048):

    frame_features_input = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm_Layer_1')(tf.keras.Input((max_seq_length, num_features)))
    
    #x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm1')(frame_features_input)
    x = TCN(input_shape=(80, num_features), nb_filters=64, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32], name='TCN_layer_01')(frame_features_input)
    x = tf.keras.layers.MaxPool1D(3, name='Stream1_MaxPool_layer_01')(x)
    x = TCN(input_shape=(80, num_features), nb_filters=128, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32], name='TCN_layer_02')(x)
    x = tf.keras.layers.MaxPool1D(3, name='Stream1_MaxPool_layer_02')(x)
    #x = TCN(input_shape=(80, num_features), nb_filters=128, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32])(x)
    #x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='Stream1_BatchNorm_Layer_1')(x)  
    x = TCN(input_shape=(80, num_features), nb_filters=64, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32], name='TCN_layer_03')(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name='Stream1_GlobalPool_layer_01')(x)
        
    #y = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001)(frame_features_input)  
    y = tf.keras.layers.LSTM(256, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.0001), name='LSTM_layer_01')(frame_features_input)
    y = tf.keras.layers.Conv1D(64, 3, input_shape=(None,max_seq_length, num_features ),padding='same', name='CNN_layer_01')(y)
    y = tf.keras.layers.MaxPool1D(3, name='Stream2_MaxPool_layer_01')(y)
    y = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='Stream2_BatchNorm_Layer_1')(y)  
    y = tf.keras.layers.LSTM(128, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.0001), name='LSTM_layer_02')(y)
    y = tf.keras.layers.Conv1D(128, 3, input_shape=(None,max_seq_length, num_features ),padding='same', name='CNN_layer_02')(y)   
    y = tf.keras.layers.MaxPool1D(3, name='Stream2_MaxPool_layer_02')(y)
    y = tf.keras.layers.LSTM(64, return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.0001), name='LSTM_layer_03')(y)
    
    #agregar rnns checar
    
    xx = tf.keras.layers.Multiply(name='layer_fusion')([x, y])
    xx = tf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=0.0001, name='BatchNorm_Layer_2')(xx)
    xx = tf.keras.layers.Dropout(0.6, name='dropout')(xx)
    xx = tf.keras.layers.Flatten()(xx)
    
    xx = tf.keras.layers.Dense(8, activation="relu")(xx)
    #xx = keras.layers.Dense(len(class_vocab), activation="softmax")(xx)
    output = tf.keras.layers.Dense(class_vocab, activation="softmax")(xx)
    rnn_model = tf.keras.Model(frame_features_input, output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

def run_experiment_model_tcn(train_ds, test_ds, epochs=30, num_features=NUM_FEATURES):
    if save_logs:
        write_logs('Loading feature extractor \n')
    
    print('TCN Loading feature extractor')
    
    MODEL = ['convnext_large','inceptionv3','vgg19','xception','mobilenetv2']
    for mod_id in MODEL:
        feature_extractor, num_features = build_feature_extractor(mod_id)
        feature_extractor.trainable = False
        # Let's take a look to see how many layers are in the base model
        print(f'base_model {mod_id}')
        
        if save_logs:
            write_logs(f'base_model {mod_id}\n')
            
        model = get_model_tcn(max_seq_length=MAX_SEQ_LENGTH, num_features=num_features)
        print("Number of layers in the base model: ", len(feature_extractor.layers))
        
        if save_logs:
            write_logs(f'Number of layers in the base model: {len(feature_extractor.layers[-1].layers)}')
            
        layer_steps = int(np.floor(len(feature_extractor.layers[-1].layers)/3))
        #for fine_tune_at in range(0, len(feature_extractor.layers[-1].layers), layer_steps):
        ### Fine-tune from this layer onwards
        ###fine_tune_at = 10
            #print(f'{mod_id} layer fine tune: {fine_tune_at}')
            
            #if (mod_id == 'convnext_large' and fine_tune_at > 293):
            #    if save_logs:
            #        write_logs(f'\n{mod_id} layer fine tune: {fine_tune_at}\n')
            # Freeze all the layers before the `fine_tune_at` layer
            
                #if (mod_id == 'xception' and fine_tune_at > 133) or mod_id != 'xception':
        fine_tune_at = len(feature_extractor.layers[-1].layers)
        for layer in feature_extractor.layers[-1].layers[:fine_tune_at]:
            layer.trainable = True

        print(f'Loading proposed model unfrozen {fine_tune_at}...')
        if save_logs:
            write_logs(f'Loading proposed model unfrozen {fine_tune_at}...\n')
    
        print('train loop')
        model = train(model, feature_extractor, train_ds=train_ds, test_ds=test_ds, batch_size=BATCH_SIZE, epochs=epochs, num_features=num_features)
    
    return model

def run_experiment_model_lstm(train_ds, test_ds, epochs=30, num_features=NUM_FEATURES):
    if save_logs:
        write_logs('Loading feature extractor\n')
    
    print('LSTM Loading feature extractor')
    
    MODEL = ['convnext_large']#,'inceptionv3','vgg19','xception','mobilenetv2']
    for mod_id in MODEL:
        feature_extractor, num_features = build_feature_extractor(mod_id)
        feature_extractor.trainable = True
        # Let's take a look to see how many layers are in the base model
        print(f'base_model {mod_id}')

        if save_logs:
            write_logs(f'base_model {mod_id}\n')
        # Let's take a look to see how many layers are in the base model
        print(feature_extractor.summary())
        
        if save_logs:
            write_logs(f'{feature_extractor.summary()}\n')
        
        layer_steps = int(np.floor(len(feature_extractor.layers[-1].layers)/3))
        model = get_model_lstm(max_seq_length=MAX_SEQ_LENGTH, num_features=num_features)
        print("Number of layers in the base model: ", len(feature_extractor.layers))
        
        
        if save_logs:
            write_logs(f'Number of layers in the base model: {len(feature_extractor.layers[-1].layers)}')
            
            
        #for fine_tune_at in range(0, len(feature_extractor.layers[-1].layers), layer_steps):
        # Fine-tune from this layer onwards
        #fine_tune_at = 10
        
            #print(f'{mod_id} layer fine tune: {fine_tune_at}')

            
            #if save_logs:
            #    write_logs(f'{mod_id} layer fine tune: {fine_tune_at} \n')
            # Freeze all the layers before the `fine_tune_at` layer
            
            #if (mod_id == 'xception' and fine_tune_at > 133) or mod_id != 'xception':
        #fine_tune_at = len(feature_extractor.layers[-1].layers)

        #for layer in feature_extractor.layers[-1].layers[:fine_tune_at]:
        #    layer.trainable = True

        #print(f'Loading proposed model unfrozen {fine_tune_at}...')
        #if save_logs:
        #    write_logs(f'Loading proposed model unfrozen {fine_tune_at}...\n')
    
        print('train loop')
        model = train(model, feature_extractor, train_ds=train_ds, test_ds=test_ds, batch_size=BATCH_SIZE, epochs=epochs, num_features=num_features)

    return model

def run_experiment_model_gru(train_ds, test_ds, epochs=30, num_features=NUM_FEATURES):
    if save_logs:
        write_logs('Loading feature extractor')
    
    print('GRU Loading feature extractor')
    
    MODEL = ['convnext_large']#,'inceptionv3','vgg19','xception','mobilenetv2']
    for mod_id in MODEL:
        feature_extractor, num_features = build_feature_extractor(mod_id)
        feature_extractor.trainable = True
        # Let's take a look to see how many layers are in the base model
        print(f'base_model {mod_id}')
        
        if save_logs:
            write_logs(f'base_model {mod_id}\n')
            
        model = get_model_gru(max_seq_length=MAX_SEQ_LENGTH, num_features=num_features)
        print(feature_extractor.summary())
        
        #layer_steps = int(np.floor(len(feature_extractor.layers[-1].layers)/3))
        print("Number of layers in the base model: ", len(feature_extractor.layers[-1].layers))
        
        
        #if save_logs:
        #    write_logs(f'Number of layers in the base model: {len(feature_extractor.layers[-1].layers)}')
            
            
        #for fine_tune_at in range(0, len(feature_extractor.layers[-1].layers), layer_steps):
            
            
        # Fine-tune from this layer onwards
        #fine_tune_at = 10
            #print(f'{mod_id} layer fine tune: {fine_tune_at}')
            
            
            #if save_logs:
            #    write_logs(f'{mod_id} layer fine tune: {fine_tune_at} \n')
            # Freeze all the layers before the `fine_tune_at` layer
            
            #if (mod_id == 'xception' and fine_tune_at > 133) or mod_id != 'xception':
            
        #fine_tune_at = len(feature_extractor.layers[-1].layers)

        #for layer in feature_extractor.layers[-1].layers[:fine_tune_at]:
        #    layer.trainable = True

        #print(f'Loading proposed model unfrozen {fine_tune_at}...')
    
        #if save_logs:
        #    write_logs(f'Loading proposed model unfrozen {fine_tune_at}...\n')
        print('train loop')
        model = train(model, feature_extractor, train_ds=train_ds, test_ds=test_ds, batch_size=BATCH_SIZE, epochs=epochs, num_features=num_features)
    
    return model

def run_experiment_model_two(train_ds, test_ds, epochs=30, num_features=NUM_FEATURES):
    if save_logs:
        write_logs('Loading feature extractor')
    
    print('TOTAL Loading feature extractor')
    MODEL = ['convnext_large']#,'inceptionv3','vgg19','xception','mobilenetv2']
    for mod_id in MODEL:
        feature_extractor, num_features = build_feature_extractor(mod_id)
        feature_extractor.trainable = True
        # Let's take a look to see how many layers are in the base model
        print(f'base_model {mod_id}')
        
        if save_logs:
            write_logs(f'base_model {mod_id}\n')
            
        model = get_model(max_seq_length=MAX_SEQ_LENGTH, num_features=num_features)
        print(feature_extractor.summary())
        
        #write_logs(feature_extractor.summary())
        #layer_steps = int(np.floor(len(feature_extractor.layers[-1].layers)/3))
        #print("Number of layers in the base model: ", len(feature_extractor.layers[-1].layers))
        
        
        #if save_logs:
        #    write_logs(f'Number of layers in the base model: {len(feature_extractor.layers[-1].layers)}')
            
        #for fine_tune_at in range(0, len(feature_extractor.layers[-1].layers), layer_steps):
        # Fine-tune from this layer onwards
        #fine_tune_at = 10
        #    print(f'{mod_id} layer fine tune: {fine_tune_at}')
            
            #if (mod_id == 'xception' and fine_tune_at > 133) or (mod_id == 'convnext_large' and fine_tune_at > 294):
            
        # Freeze all the layers before the `fine_tune_at` layer
        #    for layer in feature_extractor.layers[-1].layers[:fine_tune_at]:
        #        layer.trainable = True

        #    print(f'Loading proposed model unfrozen {fine_tune_at}...')
            
        #    if save_logs:
        #        write_logs(f'{mod_id} layer fine tune: {fine_tune_at} \n')
        
        print('train loop')
        model = train(model, feature_extractor, train_ds=train_ds, test_ds=test_ds, batch_size=BATCH_SIZE, epochs=epochs, num_features=num_features)
    
    return model

def run_experiment_one(train_ds, test_ds, epochs=30, num_features=NUM_FEATURES):
    if save_logs:
        write_logs('Loading feature extractor')
    
    print('TOTAL Loading feature extractor')
    MODEL = ['convnext_large']
    for mod_id in MODEL:
        feature_extractor, num_features = build_feature_extractor(mod_id)
        feature_extractor.trainable = False
        # Let's take a look to see how many layers are in the base model
        print(f'base_model {mod_id}')
        
        model = get_model_tcn(max_seq_length=MAX_SEQ_LENGTH, num_features=num_features)
        print("Number of layers in the base model: ", len(model.layers))
        
        for fine_tune_at in range(10, len(model.layers), 10):
        # Fine-tune from this layer onwards
        #fine_tune_at = 10

        # Freeze all the layers before the `fine_tune_at` layer
            for layer in model.layers[:fine_tune_at]:
                layer.trainable = True
    
            print(f'Loading proposed model unfrozen {fine_tune_at}...')
            
            print('train loop')
            model = train(model, feature_extractor, train_ds=train_ds, test_ds=test_ds, batch_size=BATCH_SIZE, epochs=epochs, num_features=num_features)
        
    return model


tf.keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if save_logs:
    write_logs('RGB experiments....')
    
if save_logs:
    write_logs('Loading rgb clips...\n')
    
    
#root_ds="/home/elopez/har_test/dataset/test_gt4_op_flow/"
#ds_file="/home/elopez/har_test/dataset/test_gt4_op_flow/info_95_58.csv"

root_ds = "/home/elopez/har_test/dataset/test_gt5"
info_file = "/home/elopez/har_test/dataset/test_gt5/info4cs.csv"

rgb_train_data, rgb_train_labels = get_data(root_ds, feature_extractor, split=0, num_features=NUM_FEATURES, ds_file=info_file)
rgb_test_data, rgb_test_labels = get_data(root_ds, feature_extractor, split=1, debug=False, num_features=NUM_FEATURES, ds_file=info_file)
rgb_val_data, rgb_val_labels = get_data(root_ds, feature_extractor, split=2, debug=True, num_features=NUM_FEATURES, ds_file=info_file)

print(f"Frame features in rgb_train set: {rgb_train_data[0].shape}")
print(f"Frame masks in rgb_train set: {rgb_train_data[1].shape}")

root_ds = "/home/elopez/har_test/dataset/test_gt4_op_flow"

fop_train_data, fop_train_labels = get_data(root_ds, feature_extractor, split=0, num_features=NUM_FEATURES, ds_file=info_file)
fop_test_data, fop_test_labels = get_data(root_ds, feature_extractor, split=1, debug=False, num_features=NUM_FEATURES, ds_file=info_file)
fop_val_data, fop_val_labels = get_data(root_ds, feature_extractor, split=2, debug=True, num_features=NUM_FEATURES, ds_file=info_file)

write_logs(f'TS model \n')
run_experiment_model_two(rgb_train_ds,rgb_test_ds, epochs=30)

#write_logs(f'LSTM model \n')
#run_experiment_model_lstm(rgb_train_ds,rgb_test_ds, epochs=30)
#write_logs(f'GRU model \n')
#run_experiment_model_gru(rgb_train_ds,rgb_test_ds, epochs=30)
#write_logs(f'DOC model \n')
#run_experiment_model_two(rgb_train_ds,rgb_test_ds, epochs=30)
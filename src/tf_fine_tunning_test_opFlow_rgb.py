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
from data_processing.prepare_data import load_dataset_file, load_video, label_processor

from sklearn.metrics import confusion_matrix, classification_report
from tcn import TCN, tcn_full_summary
#import seaborn as sns
os.environ["TF_USE_LEGACY_KERAS"] = "1"

IMG_SIZE = 224
MAX_SEQ_LENGTH = 16
NUM_FEATURES = 1536

#class_vocab = ['walk', 'run', 'cycle', 'fall', 'drink', 'phone']
class_vocab = 5
#class_vocab = ['walk', 'run', 'cycle', 'fall', 'drink']
#chkpnt_file = 'tmp/test_rgb_3cls_03_ene/' 90%
#chkpnt_file = 'tmp/test_rgb_4cls_03_ene/' 93.71%
chkpnt_file = 'tmp/rgbfop_3cls_8f_full_balanced_14_may_exp_01/' #'tmp/test_rgb_5cls_04_ene/' 78%
log_filename = 'two_stream/log_rgbfop_3cls_8f_full_balanced_14_may_exp_01.txt'#'log_rgb_file_05_ene.txt' anterior, falto modificar el número de clases con las que estabe entrenando #'log_rgb_file_04_ene.txt' 78%
save_logs = True


root_ds = "/home/elopez/har_test/dataset/test_gt5"
root_op_ds = "/home/elopez/har_test/dataset/test_gt4_op_flow"

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    chkpnt_file, save_weights_only=True, save_best_only=True, verbose=1
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

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
    
    
def build_feature_extractor():
    feature_extractor = tf.keras.applications.ConvNeXtLarge(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.convnext.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

def prepare_videos(x, feature_extractor, max_seq_length=30, num_features=2048, img_shape=(224, 224), min_frames=7):
    ''' 
        
    '''
    num_samples = len(x)
    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    #frame_masks = np.zeros(shape=(num_samples, max_seq_length), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, max_seq_length, num_features*2), dtype="float32"
    )
    
    # For each video.
    for idx, path in enumerate(x):
        # Gather all its frames and add a batch dimension.
        
        filepath = (root_ds + '/').encode('utf-8') + path
        fop_filepath = (root_op_ds + '/').encode('utf-8') + path
        
        if os.path.exists(filepath) and os.path.exists(fop_filepath):
            frames = load_video(filepath.decode('utf-8'), resize=img_shape)
            #print("entra")
            frames_fop = load_video(fop_filepath.decode('utf-8'), resize=img_shape)
            if len(frames) > min_frames and len(frames_fop) > min_frames:
                
                tmp_rgb = extract_features(frames,min_frames,max_seq_length,num_features,feature_extractor)
                tmp_fop = extract_features(frames_fop,min_frames,max_seq_length,num_features,feature_extractor)
                frame_features[idx,] = tf.keras.layers.concatenate([tmp_rgb,tmp_fop])#.squeeze()
                
        else:
            if not os.path.exists(filepath): print(f'File {filepath} not found')
            if not os.path.exists(fop_filepath): print(f'File {fop_filepath} not found')
            

    return frame_features

def extract_features(frames,min_frames,max_seq_length,num_features,feature_extractor):
    
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
    return temp_frame_features

def prepare_all_videos(df, max_seq_length=30, img_shape=(224, 224), min_frames=7):
    ''' 
        
    '''
    num_samples = len(df)
    video_paths = df["video_path"].values.tolist()
    labels = label_processor(df)
    frame_features = np.zeros(
        shape=(num_samples, max_seq_length, 224, 224, 3), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        print(f'Processing video {idx+1} / {num_samples}')
        if os.path.exists:
            frames = load_video(path, resize=img_shape)
            
            if len(frames) > min_frames:
                frames = frames[None, ...] 
                #print(frames.shape)             
                if frames.shape[1] >= max_seq_length:
                    frame_features[idx,] = frames[0,0:max_seq_length]
                else:
                    n_frames = frames[0]
                    missing = max_seq_length - n_frames.shape[0]
                    while missing != 0:
                        #print(f'total frames {n_frames.shape[1]} missing frames {missing}')
                        n_frames = np.concatenate((n_frames, frames[0,0:missing]))
                        missing = max_seq_length - n_frames.shape[0]
                        #print(dat.shape)
                        #print('true') if missing < n_frames.shape[1] else print('else')
                        #print(n_frames.shape)
                        #n_frames[0,] = np.concatenate((n_frames[0], n_frames[0,0:missing]))# if missing < n_frames.shape[1] else np.concatenate(n_frames[0], frames[0])
                        #print(n_frames[0].shape[1])
                        #print(f' {n_frames[0].shape[1]} n_frames[0].shape[1] < max_seq_length {n_frames[0].shape[1] < max_seq_length}')
                        
                    #print(f'n frames shape {n_frames.shape}')
                    #print(frame_features[idx].shape)
                    frame_features[idx,] = n_frames[...]
        else:
            print(f'File {path} not found')

    return frame_features, labels


def get_dataset(filepath, flow_path, split=0, ds_file='info.csv', batch_size=24, max_seq_length=30, debug=False, img_shape=(224,224)):
    df = load_dataset_file(os.path.join(filepath, ds_file), debug=debug)
    
    tmp_df = df[df['split']==split]    
    #tmp_df['op_flow_filepath'] = [os.path.join(flow_path, tmp_df.classname.loc[idx], tmp_df.filename.loc[idx]) for idx in tmp_df.index.tolist()]
    tmp_df['video_path'] = [os.path.join(tmp_df.classname.loc[idx], tmp_df.filename.loc[idx]) for idx in tmp_df.index.tolist()]
    #data, labels = prepare_all_videos(tmp_df, max_seq_length=max_seq_length, img_shape=img_shape)
    labels = label_processor(tmp_df)
    
    dataset = tf.data.Dataset.from_tensor_slices((tmp_df['video_path'], labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return dataset 


def get_fair_dataset(filepath, flow_path, split=0, ds_file='info.csv', batch_size=24, max_seq_length=30, debug=False, img_shape=(224,224)):
    df = load_dataset_file(os.path.join(filepath, ds_file), debug=debug)
    
    ## split 0 = 114
    ## split 1 = 47
    ## split 2 = 19
    
    tmp_df = df[df['split']==split]    
    #tmp_df['op_flow_filepath'] = [os.path.join(flow_path, tmp_df.classname.loc[idx], tmp_df.filename.loc[idx]) for idx in tmp_df.index.tolist()]
    tmp_df['video_path'] = [os.path.join(tmp_df.classname.loc[idx], tmp_df.filename.loc[idx]) for idx in tmp_df.index.tolist()]
    #data, labels = prepare_all_videos(tmp_df, max_seq_length=max_seq_length, img_shape=img_shape)
    labels = label_processor(tmp_df)
    
    dataset = tf.data.Dataset.from_tensor_slices((tmp_df['video_path'], labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return dataset 

@tf.function
def train_step(x, y, model):
    with tf.GradientTape() as tape:
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

def train(model, feature_extractor, train_ds, test_ds, batch_size=24, epochs = 2):
    best_acc =0
    for epoch in range(epochs):
        
        print("\nStart of epoch %d" % (epoch,))
        if save_logs:
            write_logs("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                #print(x_batch_train)
                x_data = prepare_videos(x_batch_train.numpy(), max_seq_length=MAX_SEQ_LENGTH, feature_extractor=feature_extractor, num_features=NUM_FEATURES)
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
                    write_logs("\nTraining loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
        train_acc = train_acc_metric.result()
        
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        if save_logs:
            write_logs("\nTraining acc over epoch: %.4f" % (float(train_acc),))
        train_acc_metric.reset_states()
        
        for x_batch_test, y_batch_test in test_ds:
            x_test = prepare_videos(x_batch_test.numpy(), max_seq_length=MAX_SEQ_LENGTH, feature_extractor=feature_extractor, num_features=NUM_FEATURES)
            test_step(x_test, y_batch_test, model)
            
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_weights(chkpnt_file.format(epoch=epoch))

        if save_logs:
            write_logs("\nValidation acc: %.4f" % (float(val_acc),))
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        if save_logs:
            write_logs("\nTime taken: %.2fs" % (time.time() - start_time))
    return model


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

def run_experiment(train_ds, test_ds, epochs=30, num_features=NUM_FEATURES):
    if save_logs:
        write_logs('Loading feature extractor')
    
    print('Loading feature extractor')
    feature_extractor = build_feature_extractor()
    
    print('Loading proposed model...')
    model = get_model(max_seq_length=MAX_SEQ_LENGTH, num_features=num_features*2)
    
    print('Merging models...')
    #model = tf.keras.Sequential([feature_extractor, top_model])class_vocab
    #history = model.fit(/home/elopez/
    #    train,
    #    labels[0],
    #    validation_split=0.3,
    #    epochs=epochs,
    #   callbacks=[checkpoint],
    #)
    
    print('train loop')
    model = train(model, feature_extractor, train_ds=train_ds, test_ds=test_ds, batch_size=64, epochs=epochs)
    
    #if os.path.exists(filepath):
    #    model.load_weights(filepath)
        
    #_, accuracy = model.evaluate(test, labels[1])
    
    #if save_logs:
    #    write_logs(round(accuracy * 100, 2))
        
    #print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model


if __name__ == '__main__':
    
    
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
        write_logs('\n rgb + optical flow experiments....\n')
        
        write_logs('\n full fine tunning....\n')
    #if save_logs:
    #    write_logs('Loading optical flow pose clips...\n')
    
    #root_ds = "/home/elopez/har_test/dataset/test_gt5"
    ds_file = "/home/elopez/har_test/dataset/info_files/info_3cls_balanced.csv"
    #root_op_ds = "/home/elopez/har_test/dataset/test_gt4_op_flow"
    
    rgb_train_ds = get_dataset(root_ds, root_op_ds, split=0, ds_file=ds_file)
    rgb_test_ds = get_dataset(root_ds, root_op_ds, split=1, debug=False, ds_file=ds_file)
    rgb_val_ds = get_dataset(root_ds, root_op_ds, split=2, debug=True, ds_file=ds_file)
    
    #if save_logs:
    #    write_logs(f"Frame features in rgb_train set: {rgb_train_data.shape}")
    
    #      write_logs('Loading opflow_pose clips......\n')
    
    

    
    run_experiment(rgb_train_ds,rgb_test_ds, epochs=20)
    #run_experiment(fop_train_ds,fop_test_ds, epochs=50)

    # if save_logs:
    #     write_logs(f"Frame features in fop_train set: {fop_train_data.shape}\n")    
    
    #     write_logs('Data concatenation...\n')
        
    # rgb_train_data = tf.keras.layers.concatenate([rgb_train_data, fop_train_data])
    # print(f'train size {rgb_train_data.shape} sim_merged_train_features {rgb_train_data.shape}')
    
    # rgb_test_data = tf.keras.layers.concatenate([rgb_test_data, fop_test_data])
    # rgb_val_data = tf.keras.layers.concatenate([rgb_val_data, fop_val_data])
    
    # if save_logs:
    #     write_logs(f'train size {rgb_train_data.shape} sim_merged_train_features {rgb_train_data.shape}\n')
    
    #     write_logs('running experiment...')
 

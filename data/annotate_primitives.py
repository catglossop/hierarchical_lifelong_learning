import pickle as pkl
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio


MIN_STEPS = 5
MAX_STEPS = 10
DATASET_PATH = "/Users/catherineglossop/LLLwL/gnm_dataset/sacson"
VISUALIZE = False
DEBUG = False

varied_forward = [
    "Move forward",
    "Proceed straight",
    "Continue ahead",
    "Keep going forward",
    "Advance",
    "Go straight on",
    "Head straight",
    "March forward",
    "Push forward",
    "Progress forward",
    "Forge ahead",
    "Maintain your course",
    "Stay on track",
    "Keep moving ahead",
    "Drive straight ahead",
    "Press on",
    "Plow ahead",
    "Move ahead",
    "Continue on your path",
    "Sustain your direction"
]
varied_right = [
    "Make a right turn",
    "Veer to the right",
    "Head rightward",
    "Take a right here",
    "Go right at the next opportunity",
    "Bear right",
    "Swing to the right",
    "Proceed to the right",
    "Angle right",
    "Shift right",
    "Rotate right",
    "Pivot to the right",
    "Steer right",
    "Divert to the right",
    "Bank right",
    "Curve right",
    "Move to the right side",
    "Navigate right",
    "Aim right",
    "Adjust your path to the right"
]
varied_left = [
    "Make a left turn",
    "Veer to the left",
    "Head leftward",
    "Take a left here",
    "Go left at the next opportunity",
    "Bear left",
    "Swing to the left",
    "Proceed to the left",
    "Angle left",
    "Shift left",
    "Rotate left",
    "Pivot to the left",
    "Steer left",
    "Divert to the left",
    "Bank left",
    "Curve left",
    "Move to the left side",
    "Navigate left",
    "Aim left",
    "Adjust your path to the left"
]

varied_stop = [
    "cease",
    "halt",
    "desist",
    "terminate",
    "end",
    "quit",
    "suspend",
    "discontinue",
    "abandon",
    "forbear",
    "pause",
    "ceasefire",
    "standstill",
    "break off",
    "conclude",
    "finish",
    "terminate",
    "cease and desist",
    "bring to a halt",
    "put an end to"
]

folder_names = []
print("LOADING DATA ...")
for name in os.listdir(DATASET_PATH):
        if os.path.isdir(os.path.join(DATASET_PATH,name)):
            try:
                folder_names.append(os.path.join(DATASET_PATH, name))
            except Exception as e:
                print(f'Encountered error processing folder {name} with error {e}')

def get_yaw_delta(yaw_reshape):
    yaw_delta = yaw_reshape[:,-1] - yaw_reshape[:,0]
    yaw_delta_sign = np.where(yaw_delta >= np.pi, -1, 0)
    yaw_delta_sign = np.where(yaw_delta < -np.pi, 1, yaw_delta_sign)
    yaw_delta = yaw_delta + yaw_delta_sign*2*np.pi
    return yaw_delta

yaw_avgs = []
yaw_stds = []
traj_len_hist = {}
yaw = np.empty((0,1))
print("Getting dataset action stats ...")
for chunk_size in range(MIN_STEPS, MAX_STEPS):
    yaw_avg_i = []
    yaw_std_i = []
    for folder in folder_names:

        with open(os.path.join(folder, 'traj_data.pkl'), 'rb') as f:
            data = pkl.load(f)
            yaw = data["yaw"]
            if yaw.shape[0] >= chunk_size:
                if chunk_size in traj_len_hist.keys():
                    traj_len_hist[chunk_size] += 1
                else:
                    traj_len_hist[chunk_size] = 1
                if len(data["yaw"].shape) < 2:
                    yaw = np.expand_dims(yaw, axis=-1)
                yaw_i = yaw
                if yaw.shape[0] % chunk_size != 0:
                    yaw_i = yaw_i[:chunk_size*(yaw.shape[0]//chunk_size)]
                yaw_reshape = yaw_i.reshape(-1,chunk_size)
                yaw_delta = np.abs(get_yaw_delta(yaw_reshape))
                yaw_avg = yaw_delta.mean()
                yaw_std = yaw_delta.std()
                yaw_avg_i.append(yaw_avg)
                yaw_std_i.append(yaw_std)

    yaw_avgs.append(np.array(yaw_avg_i).mean())
    yaw_stds.append(np.array(yaw_std_i).mean())

yaw_avgs = np.asarray(yaw_avgs)
yaw_avg_normed = yaw_avgs/np.max(yaw_avgs)
yaw_stds = np.asarray(yaw_stds)
yaw_std_normed = yaw_stds/np.max(yaw_stds)

J = np.square(yaw_avg_normed) - np.square(yaw_std_normed)
CHUNK_SIZE = MIN_STEPS + np.argmax(J)
TURN_THRESHOLD = yaw_avgs[CHUNK_SIZE-MIN_STEPS]
# TURN_THRESHOLD = 0.5

fig, ax = plt.subplots(1,2)

ax[0].errorbar(np.arange(MIN_STEPS,MAX_STEPS), yaw_avgs, yaw_stds)
ax[1].bar(traj_len_hist.keys(), traj_len_hist.values())
plt.show()

position_dists = []
for folder in folder_names:
    with open(os.path.join(folder, 'traj_data.pkl'), 'rb') as f:
        # Load data 
        data = pkl.load(f)
        position = data["position"]
        yaw = data["yaw"]
        if position.shape[0] >= CHUNK_SIZE:
            if position.shape[0] % CHUNK_SIZE != 0:
                position = position[:CHUNK_SIZE*(position.shape[0]//CHUNK_SIZE)]
                yaw = yaw[:CHUNK_SIZE*(position.shape[0]//CHUNK_SIZE)]
            position_reshape = position.reshape(-1, CHUNK_SIZE, 2)
            yaw_reshape = yaw.reshape(-1, CHUNK_SIZE)
            yaw_delta = np.abs(get_yaw_delta(yaw_reshape))
            yaw_mask = np.argwhere(np.abs(yaw_delta) < TURN_THRESHOLD)
            position_reshape = position_reshape[yaw_mask, :,:].squeeze(1)
            if position_reshape.shape[0] >= 1:
                position_delta = position_reshape[:,-1,:] - position_reshape[:,0,:] 
                position_dist = np.sqrt(np.sum(np.square(position_delta), axis=-1))
                position_dists.append(position_dist.mean())

position_dists = np.array(position_dists)
DIST_MEAN = position_dists.mean()
DIST_MAX = np.max(position_dists)
DIST_MIN = np.min(position_dists)
DIST_STD = np.std(position_dists)

FORWARD_THRESHOLD = DIST_MEAN 
STOP_THRESHOLD = 0.2

print("DATASET STATS: ")
print("-----------------")
print(f"Optimal chunksize: {CHUNK_SIZE}")
print(f"TURN THRESHOLD: {TURN_THRESHOLD}")
print(f"STOP THRESHOLD: {STOP_THRESHOLD}")
print(f"DIST MIN: {DIST_MIN}")
print(f"DIST STD: {DIST_STD}")
print(f"DIST_MEAN: {FORWARD_THRESHOLD}")

            

base_instructions = ["Turn left", "Turn right", "Go forward", "Stop"]


def compute_lang_instruc(traj, chunk_size, yaw_threshold, pos_threshold):

    yaw = traj["yaw"]
    pos = traj["position"]

    num_chunks = len(yaw)//chunk_size 
    yaw_chunks = np.split(yaw, num_chunks)
    pos_chunks = np.split(pos, num_chunks)
    image_chunks = np.split(traj["obs"], num_chunks)

    samples_out = []
    
    for yaw_chunk, pos_chunk, image_chunk in zip(yaw_chunks, pos_chunks, image_chunks): 

        yaw_delta = float(get_yaw_delta(yaw_chunk).squeeze())
        pos_delta = np.sqrt(np.sum(np.square(pos_chunk[-1,:] - pos_chunk[0,:]), axis=-1))

        if yaw_delta > yaw_threshold:
            lang = base_instructions[0]
            varied_lang = random.choice(varied_left)
        elif yaw_delta < -yaw_threshold:
            lang = base_instructions[1]
            varied_lang = random.choice(varied_right)
        else: 
            if pos_delta > pos_threshold:
                lang = base_instructions[2]
                varied_lang = random.choice(varied_forward)
            else:
                lang = base_instructions[3]
                varied_lang = random.choice(varied_stop)
        sample = {}
        sample["obs"] = image_chunk
        sample["lang"] = lang
        sample["varied_lang"] = varied_lang

        samples_out.append(sample)
    
    data = tf.concat(samples_out, axis=0)
    print(data.size())
    return data

def relabel_primitives(dataset, chunk_size, yaw_threshold, pos_threshold):
    ''' Preprocess the tf dataset into chunks with language instructions'''
    # TODO: believe dataset will now be list of trajectories 
    # Send traj to be mapped to chunks and instructions
    dataset = dataset.map(
        partial(compute_lang_instruc, chunk_size=chunk_size, yaw_threshold=yaw_threshold, pos_threshold=pos_threshold), 
        num_parallel_calls=None
    )
    print(dataset.size())

    return dataset

def relabel_vlm(dataset): 
    ''' Relabel the dataset with the VLM instructions'''
    # Should have available list of skills ? 
    # Otherwise just use VLM to get an idea of what happened
    pass

def get_language_instructions(yaw, pos):
    language_instructions = []
    varied_language_instructions = []

    CHUNKS = len(yaw)//CHUNK_SIZE if len(yaw)%CHUNK_SIZE == 0 else len(yaw)//CHUNK_SIZE + 1
    for i in range(CHUNKS):
        end = (i+1)*CHUNK_SIZE
        if (i+1)*CHUNK_SIZE >= yaw.shape[0]:
            end = -1
        yaw_reshape = np.expand_dims(yaw[i*CHUNK_SIZE:end],0)
        if i*CHUNK_SIZE == len(yaw)-1:
            yaw_reshape = np.expand_dims(yaw[i*CHUNK_SIZE:],0)
        yaw_delta = float(get_yaw_delta(yaw_reshape).squeeze())
        pos_delta = np.sqrt(np.sum(np.square(pos[end,:] - pos[i*CHUNK_SIZE,:]), axis=-1))
        if DEBUG:
            print("Yaw delta: ", yaw_delta)
            print("Yaw: ", yaw_reshape)
            print("Yaw start: ", yaw[i*CHUNK_SIZE])
            print("Yaw end: ", yaw[end])
        if yaw_delta > TURN_THRESHOLD:
            if DEBUG:
                print("Result is turn left")
            language_instructions.append(base_instructions[0]) 
            varied_language_instructions.append(random.choice(varied_left))
        elif yaw_delta < -TURN_THRESHOLD:
            if DEBUG:
                print("Result is turn right")
            language_instructions.append(base_instructions[1]) 
            varied_language_instructions.append(random.choice(varied_right))
        else:
            if DEBUG:
                print("Pos delta: ", pos_delta)
            if pos_delta > STOP_THRESHOLD:
                if DEBUG:
                    print("Result is go forward")
                language_instructions.append(base_instructions[2]) 
                varied_language_instructions.append(random.choice(varied_forward))
            elif pos_delta < STOP_THRESHOLD:
                if DEBUG:
                    print("Result is stop")
                language_instructions.append(base_instructions[3]) 
                varied_language_instructions.append(random.choice(varied_stop))
        
        
        if DEBUG:
            fig, ax = plt.subplots(1,CHUNK_SIZE, figsize=(15, 15))
            for j in range(yaw_reshape.shape[1]):
                print("image number ", i*CHUNK_SIZE + j)
                image = iio.imread(os.path.join(folder, f"{i*CHUNK_SIZE + j}.jpg"))
                ax[j].imshow(image)
                ax[j].set_title(language_instructions[-1] + "\n " + str(np.round(yaw_delta, 2)))
            plt.show()
            
    return language_instructions, varied_language_instructions    

for folder in folder_names:
    if DEBUG:
        print(f'Processing folder {folder}')
    with open(os.path.join(folder, 'traj_data.pkl'), 'rb') as f:
        data = pkl.load(f)
        language_instructions, varied_language_instructions = get_language_instructions(data['yaw'], data['position'])
        data['language_instructions'] = language_instructions
        data['varied_language_instructions'] = varied_language_instructions
        data['chunk_size'] = CHUNK_SIZE
        
        # Save the language data
        with open(os.path.join(folder, 'traj_data_language.pkl'), 'wb') as g:
            pkl.dump(data, g)


if VISUALIZE: 

    folder = random.choice(folder_names)
    long_enough = False
    keep_going = True
    while keep_going: 
        while not long_enough:
            with open(os.path.join(folder, "traj_data_language.pkl"), 'rb') as f: 
                data = pkl.load(f)
                if len(data["language_instructions"]) >= 5*CHUNK_SIZE:
                    long_enough = True
                else:
                    folder = random.choice(folder_names)
                language_instructions = data["language_instructions"]
                yaw = data["yaw"]
        yaw_reshape = yaw[:CHUNK_SIZE*5].reshape(-1,CHUNK_SIZE)
        yaw_delta = get_yaw_delta(yaw_reshape)
        breakpoint()
        fig, ax = plt.subplots(5,CHUNK_SIZE, figsize=(15, 15))
        for i in range(5):
            print("YAWS:", yaw_reshape[i,:])
            get_language_instructions(yaw_reshape[i,:], data["position"])
            print("YAW DELTA:", yaw_delta[i])
            for j in range(CHUNK_SIZE):

                image = iio.imread(os.path.join(folder, f"{i*5 + j}.jpg"))
                ax[i, j].imshow(image)
                ax[i, j].set_title(language_instructions[i] + "\n " + str(round(yaw_delta[i], 2)))
                
        plt.show()
        
        keep_going = input("Keep going? (y/n): ") == "y"
        long_enough = False

        plt.close()







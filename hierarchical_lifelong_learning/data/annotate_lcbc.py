import pickle as pkl
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio
import shutil


MIN_STEPS = 5
MAX_STEPS = 10
DATASET = "go_stanford2"
DATASET_PATH = f"/home/noam/LLLwL/datasets/gnm_dataset/{DATASET}"
OVERWRITE = True
NEW_DATASET_PATH = f"/home/noam/LLLwL/datasets/lang_dataset/{DATASET}"
if OVERWRITE:
    os.makedirs(NEW_DATASET_PATH, exist_ok=True)
else:
    os.makedirs(NEW_DATASET_PATH, exist_ok=False)
VISUALIZE = False
DEBUG = False

base_instructions = ["Turn left", "Turn right", "Go forward", "Stop"]
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

CHUNK_SIZE = 10
TURN_THRESHOLD = np.pi/4
STOP_THRESHOLD = 0.2

def load_data():    
    folder_names = []
    print("LOADING DATA ...")
    for name in os.listdir(DATASET_PATH):
        if os.path.isdir(os.path.join(DATASET_PATH,name)):
            try:
                folder_names.append(os.path.join(DATASET_PATH, name))
            except Exception as e:
                print(f'Encountered error processing folder {name} with error {e}')
    return folder_names

def get_yaw_delta(yaw_reshape):
    yaw_delta = yaw_reshape[:,-1] - yaw_reshape[:,0]
    yaw_delta_sign = np.where(yaw_delta >= np.pi, -1, 0)
    yaw_delta_sign = np.where(yaw_delta < -np.pi, 1, yaw_delta_sign)
    yaw_delta = yaw_delta + yaw_delta_sign*2*np.pi
    return yaw_delta

def get_language_instructions(yaw, pos):
    language_instructions = []
    varied_language_instructions = []

    chunks = len(yaw)//CHUNK_SIZE if len(yaw)%CHUNK_SIZE == 0 else len(yaw)//CHUNK_SIZE + 1
    for i in range(chunks):
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
            
    return language_instructions, varied_language_instructions    

def main():
    folder_names = load_data()
    for folder in folder_names:
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

        long_enough = False
        keep_going = True
        while keep_going: 
            while not long_enough:
                folder = random.choice(folder_names)
                with open(os.path.join(folder, "traj_data_language.pkl"), 'rb') as f: 
                    data = pkl.load(f)
                    language_instructions = data["language_instructions"]
                    print("Before: ", len(language_instructions))
                    language_instructions = language_instructions[:CHUNK_SIZE*(len(language_instructions)//CHUNK_SIZE)]
                    print("After: ", len(language_instructions))
                    if base_instructions[0] in language_instructions and base_instructions[1] in language_instructions and base_instructions[2] in language_instructions and base_instructions[3] in language_instructions:
                        print(language_instructions)
                        long_enough = True
                        language_instructions = data["language_instructions"]
                        yaw = data["yaw"]
                        pos = data["position"]

            tl_index = np.argwhere(np.array(language_instructions) == base_instructions[0])[0].squeeze()
            tr_index = np.argwhere(np.array(language_instructions) == base_instructions[1])[0].squeeze()
            gf_index = np.argwhere(np.array(language_instructions) == base_instructions[2])[0].squeeze()
            st_index = np.argwhere(np.array(language_instructions) == base_instructions[3])[0].squeeze()

            print(tl_index)
            print(tr_index)
            print(gf_index)
            print(st_index)

            yaw_reshape = yaw[:CHUNK_SIZE*(yaw.shape[0]//CHUNK_SIZE)].reshape(-1,CHUNK_SIZE)
            pos_reshape = pos[:CHUNK_SIZE*(pos.shape[0]//CHUNK_SIZE)].reshape(-1,CHUNK_SIZE,2)

            yaw_delta = get_yaw_delta(yaw_reshape)
            pos_delta = np.sqrt(np.sum(np.square(pos_reshape[:,-1,:] - pos_reshape[:,0,:]), axis=-1))
            fig, ax = plt.subplots(len(base_instructions), CHUNK_SIZE, figsize=(4, 10))
            # plot the images 
            for j in range(CHUNK_SIZE):
                image = iio.imread(os.path.join(folder, f"{tl_index*CHUNK_SIZE + j}.jpg"))
                ax[0, j].imshow(image)
                ax[0, j].set_title("Turn left" + "\n " + "YD: " + str(round(yaw_delta[tl_index], 2)) + "\n" + "PD: " + str(round(pos_delta[tl_index], 2)))

                image = iio.imread(os.path.join(folder, f"{tr_index*CHUNK_SIZE + j}.jpg"))
                ax[1, j].imshow(image)
                ax[1, j].set_title("Turn right" + "\n " + "YD: " + str(round(yaw_delta[tr_index], 2)) + "\n" + "PD: " + str(round(pos_delta[tr_index], 2)))

                image = iio.imread(os.path.join(folder, f"{gf_index*CHUNK_SIZE + j}.jpg"))
                ax[2, j].imshow(image)
                ax[2, j].set_title("Go forward" + "\n " + "YD: " + str(round(yaw_delta[gf_index], 2)) + "\n" + "PD: " + str(round(pos_delta[gf_index], 2)))

                image = iio.imread(os.path.join(folder, f"{st_index*CHUNK_SIZE + j}.jpg"))
                ax[3, j].imshow(image)
                ax[3, j].set_title("Stop" + "\n " + "YD: " + str(round(yaw_delta[st_index], 2)) + "\n" + "PD: " + str(round(pos_delta[st_index], 2)))
            
            plt.show()
                        
                
            keep_going = input("Keep going? (y/n): ") == "y"
            long_enough = False

            plt.close()
    
    new_traj_idx = 0
    for folder in folder_names:
        with open(os.path.join(folder, 'traj_data_language.pkl'), 'rb') as f:
            lang_data = pkl.load(f)
        with open(os.path.join(folder, 'traj_data.pkl'), 'rb') as f:
            data = pkl.load(f)
            yaw = data['yaw']
            pos = data['position']
            language_instructions = lang_data['language_instructions']
            varied_language_instructions = lang_data['varied_language_instructions']
        
        chunks = len(yaw)//CHUNK_SIZE if len(yaw)%CHUNK_SIZE == 0 else len(yaw)//CHUNK_SIZE + 1
        for i in range(chunks):
            end = (i+1)*CHUNK_SIZE
            if (i+1)*CHUNK_SIZE >= yaw.shape[0]:
                end = -1
            traj_yaw = yaw[i*CHUNK_SIZE:end]
            traj_pos = pos[i*CHUNK_SIZE:end]
            traj_lang = language_instructions[i]
            traj_varied_lang = varied_language_instructions[i]

            os.makedirs(os.path.join(NEW_DATASET_PATH, f"{DATASET}_traj_{new_traj_idx}"), exist_ok=True)
            with open(os.path.join(NEW_DATASET_PATH, f"{DATASET}_traj_{new_traj_idx}", "traj_data.pkl"), 'wb') as g:
                new_data = {"yaw": traj_yaw, "position": traj_pos, "language_instructions": traj_lang, "varied_language_instructions": traj_varied_lang}
                pkl.dump(new_data, g)
            
            if (i+1)*CHUNK_SIZE >= yaw.shape[0]:
                end = len(yaw)
            for j in range(i*CHUNK_SIZE, end):
                shutil.copy(os.path.join(folder, f"{j}.jpg"), os.path.join(NEW_DATASET_PATH, f"{DATASET}_traj_{new_traj_idx}", f"{j - i*CHUNK_SIZE}.jpg"))
        
            new_traj_idx += 1



if __name__ == "__main__":
    main()







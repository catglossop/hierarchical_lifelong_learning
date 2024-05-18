#!/bin/bash

# Create a new tmux session
session_name="hl_roomba_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves

# Run the create_topoplan.py script with command line args in the second pane
tmux select-pane -t 0
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "conda activate lifelong" Enter
tmux send-keys "python ~/create_ws/src/hierarchical_learning/hierarchical-lifelong-learning/create_topomap.py --dir $1 --dt 1" Enter

# Change the directory to ../topomaps/bags and run the rosbag play command in the third pane
tmux select-pane -t 1
tmux send-keys "mkdir -p topomaps/bags" Enter
tmux send-keys "cd topomaps/bags" Enter
tmux send-keys "ros2 bag play -r 0.5 $2" # feel free to change the playback rate to change the edge length in the graph

# Attach to the tmux session
tmux -2 attach-session -t $session_name

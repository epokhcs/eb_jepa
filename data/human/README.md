=========================== Dataset Description ===========================

We provide a large-scale, high-quality dataset of human actions with simultaneously recorded eye movements while humans play Atari video games. The dataset consists of 117 hours of gameplay data from a diverse set of 20 games, with 8 million action demonstrations and 328 million gaze samples. We introduce a novel form of gameplay, in which the human plays in a semi-frame-by-frame manner. This leads to near-optimal game decisions and game scores that are comparable or better than known human records. For every game frame, its corresponding image frame, the human keystroke action, the reaction time to make that action, the gaze positions, and immediate reward returned by the environment were recorded.

 

Q & A: Why frame-by-frame game mode?

Resolving state-action mismatch: Closed-loop human visuomotor reaction time is around 250-300 milliseconds. Therefore, during gameplay, state (image) and action that are simultaneously recorded at time step t could be mismatched. Action at time t could be intended for a state 250-300ms ago. This effect causes a serious issue for supervised learning algorithms, since label at and input st are no longer matched. Frame-by-frame game play ensures states and actions are matched at every timestep.

Maximizing human performance: Frame-by-frame mode makes gameplay more relaxing and reduces fatigue, which could normally result in blinking and would corrupt eye-tracking data. More importantly, this design reduces sub-optimal decisions caused by inattentive blindness.

Highlighting critical states that require multiple eye movements: Human decision time and all eye movements were recorded at every frame. The states that could lead to a large reward or penalty, or the ones that require sophisticated planning, will take longer and require multiple eye movements for the player to make a decision. Stopping gameplay means that the observer can use eye-movements to resolve complex situations. This is important because if the algorithm is going to learn from eye-movements it must contain all “relevant” eye-movements.

 

============================ Readme ============================

1. meta_data.csv: meta data for the dataset., including:

GameName: String. Game name. e.g., “alien” indicates the trial is collected for game Alien (15 min time limit). “alien_highscore” is the trajectory collected from the best player’s highest score (2 hour limit). See dataset description paper for details.

trial_id: Integer. One can use this number to locate the associated .tar.bz2 file and label file.

subject_id: Char. Human subject identifiers.

load_trial: Integer. 0 indicates that the game starts from scratch. If this field is non-zero, it means that the current trial continues from a saved trial. The number indicates the trial number to look for.

highest_score: Integer. The highest game score obtained from this trial.

total_frame: Number of image frames in the .tar.bz2 repository.

total_game_play_time: Integer. game time in ms. 

total_episode: Integer. number of episodes in the current trial. An episode terminates when all lives are consumed.

avg_error: Float. Average eye-tracking validation error at the end of each trial in visual degree (1 visual degree = 1.44 cm in our experiment). See our paper for the calibration/validation process.

max_error: Float. Max eye-tracking validation error. 

low_sample_rate: Percentage. Percentage of frames with less than 10 gaze samples. The most common reason for this is blinking.

frame_averaging: Boolean. The game engine allows one to turn this on or off. When turning on (TRUE), two consecutive frames are averaged, this alleviates screen flickering in some games.

fps: Integer. Frame per second when an action key is held down.

 

2. [game_name].zip files: these include data for each game, including:

*.tar.bz2 files: contains game image frames. The filename indicates its trial number.

*.txt files: label file for each trial, including:

frame_id: String. The ID of a frame, can be used to locate the corresponding image frame in .tar.bz2 file.

episode_id: Integer (not available for some trials). Episode number, starting from 0 for each trial. A trial could contain a single trial or multiple trials.

score: Integer (not available for some trials). Current game score for that frame.

duration(ms): Integer. Time elapsed until the human player made a decision. 

unclipped_reward: Integer. Immediate reward returned by the game engine.

action: Integer. See action_enums.txt for the mapping. This is consistent with the Arcade Learning Environment setup.

gaze_positions: Null/A list of integers: x0,y0,x1,y1,...,xn,yn. Gaze positions for the current frame. Could be null if no gaze. (0,0) is the top-left corner. x: horizontal axis. y: vertical.

 

3.  action_enums.txt: contains integer to action mapping defined by the Arcade Learning Environment. 


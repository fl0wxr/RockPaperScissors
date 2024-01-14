# Rock Paper Scissors

A setup where an intelligent agent learns how to play a game of rock paper scissors based on visual stimuli and based on an external dataset that parameterizes the agent's environment.

Basic step rules

- Rock wins scissors
- Scissors wins paper
- Paper wins rock
- Rock wins scissors

In each step, the agent receives an input image of a hand that tries to depict either rock, paper or scissors, and then it has to figure out and depending on its observation (i.e. an image), it produces its corresponding action.

The game is played as follows: A player always plays first in one step, and after them the agent has to play based only on the image it receives from that player. In each step, the agent bets 1 euro, and a sum is returned depending of if they win or not. The following 3 scenarios are all step possibilities (depending on the agent's decision)

- Win round -> Returns: 2 euro
- Tie round -> Returns: 1 euro
- Loss round -> Returns: -1 euro

The number of steps before the round terminates is set to 3. Hence 3 steps per round. With a maximum return of 6 euros and maximum loss of 3 euros.

## Selected Setup

The model that was selected is based on a CNN trained utilizing the PyTorch backend implementation of the PPO algorithm. The CNN architecture is more suitable for this task as it revolves around an image modality. I have tried using other variations of policy CNNs, however due to the high training time, I have decided to accept one CNN. I have avoided using dimensionality reduction algorithms due to the fact that transformations such as PCA or LDA usually seem to worsen image based trainings, probably because they are not designed to retain spacial information. Hence I have decided to simply resize (i.e. use 2D interpolation) for the image preprocessing as it made more sense to me and for simplicity. With the intention of preserving numerical stability during training, I have normalized the image pixel values in the $[0,1]$ interval.

## Training Details

The total number of training epochs is set to 20, with the training taking up to 109 minutes to complete. On top of that, by generating synthetic images during training, I have expanded the train set in a valid way that allows the policy network to capture more relevant patterns.

## System Specifications

The model training and evaluation were performed on a system with the following specifications:

- **OS**: Ubuntu 22.04.3 LTS
- **CPU**: Intel Core i5 12500H
- **GPU**: NVIDIA RTX 4060
- **Memory**: 38.9 GiB RAM

## Model Evaluation

The resulting trained model achieves a 0.915 test accuracy (Counting as true positives only the wins-per-one-step), where my proposed baseline is set to be 2/3 which is the accuracy of a random agent. 5.547 average reward per game (3 steps/rounds) with the conditions of
- Win round -> Returns: 2 euro
- Tie round -> Returns: 1 euro
- Loss round -> Returns: -1 euro
In this case, the selected baseline for that metric is ((2+1-1)/3)*(n_rounds) = 2 EUR.

One can intuitively speculate that performance drops notably when external images of the relevant hand formations (i.e. rock, scissors, paper hand formations) are inserted into the model as inputs. My evidence are the images tested from the `./small_test_sample` directory. This is obviously due to overfitting tendencies of the agent's policy model, as the model has not seen images with different backgrounds, many other different hands, wrists with bracelets or watches etc. Only by considering the fact that the training is limited on green backgrounds, we expect the model to behave in a biased way when the background is white for example.

## Potential Improvement Directions

Hence the train set could be expanded in a way so that images would include new objects and variations such as the ones mentioned in the previous section. Additional augmentation would also be another cheap but nevertheless an effective alternative to boost performance. Also an edge detector or an image segmenter that splits the image in a hand vs background would significantly assist the agent to process its observations more neatly.
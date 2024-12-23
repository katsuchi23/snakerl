import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from env import SnakeGame
from collections import deque
import random
import pygame

# Environment Setup
env = SnakeGame(speed = 50)

# Model Construction
class Model(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Model, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.features = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )

    def forward(self, x):
        x = self.features(x)
        return x
    

# Trainer class and training loop
class QTrainer:

    def __init__(self, model):
        self.lr = 0.001
        self.max_moves = 200
        self.epochs = 1000
        self.episode = 100
        self.model = model
        self.gamma = 0.9
        self.max_epsilon = 0
        self.min_epsilon = 0
        self.memory = deque(maxlen=200_000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.criterion = nn.HuberLoss()
        self.batch = 2048

    def get_action(self, state, n_games): 
        action = [0,0,0]
        self.model.train()

        # Make epsilon reduce in 80% of the game
        total_game = 0.8 * (self.episode * self.epochs)
        decay_rate = (self.max_epsilon - self.min_epsilon) / total_game
        epsilon = self.max_epsilon - (decay_rate * n_games)

        if torch.rand(1).item() < epsilon:
            random_number = torch.randint(0, 3, (1,)).item()
            action[random_number] = 1
            return action
        
        states = np.array(state)
        states = torch.tensor(states, dtype = torch.float32).to(device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(states)
        # print(output)

        self.model.train()
        idx = torch.argmax(output).item()
        action[idx] = 1
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch):
        if len(self.memory) > batch:
            batchs = random.sample(self.memory, batch)
        else:
            batchs = self.memory
        
        states, actions, rewards, next_states, dones = zip(*batchs)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            rewards = torch.unsqueeze(rewards, 0)
            dones = torch.unsqueeze(dones, 0)
        
        # Get current Q values
        current_q_values = self.model(states)
        
        # Get action index
        action_idx = torch.tensor([actions.index(1)], dtype=torch.long).to(device)
        
        # Get Q value for taken action
        current_q_value = current_q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        
        # Compute target Q value
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_value, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def train_short_term(self, moves):
        if len(self.memory) == 0:
            print("No memory to train from. Skipping short-term training.")
            return 0
        
        # Sample batch
        if len(self.memory) > moves:
            batch = random.sample(self.memory, moves)
        else:
            batch = self.memory
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        # Convert actions to indices
        action_indices = torch.tensor([list(action).index(1) for action in actions], 
                                    dtype=torch.long).to(device)
        
        # Get current Q values
        current_q_values = self.model(states)
        
        # Get Q values for taken actions
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def train_long_term(self):
        # Sample batch
        if len(self.memory) > self.batch:
            batch = random.sample(self.memory, self.batch)
        else:
            batch = self.memory
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        # Convert actions to indices
        action_indices = torch.tensor([list(action).index(1) for action in actions], 
                                    dtype=torch.long).to(device)
        
        # Get current Q values
        current_q_values = self.model(states)
        
        # Get Q values for taken actions
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(11,3).to(device)
    state_dict = torch.load('snake_model_q_learning.pth')
    model.load_state_dict(state_dict)
    pygame.init()
    Agent = QTrainer(model)
    batch_size = 64
    total_game = 0
    
    for epoch in range(Agent.epochs):
        for episode in range(Agent.episode):
            total_game += 1
            moves = 0
            step = 0
            state = env.reset()  # get initial state
            total_reward = 0 # initialize reward and score per game
            scores = 0
            
            while moves < Agent.max_moves:
                moves += 1
                step += 1       
                current_action = Agent.get_action(state, total_game)  # renamed to current_action
                reward, done, score = env.step(current_action)

                # Update the score and rewatd
                new_scores = score # disini score itu sudah total score accumulate sepanjang 1 episode makanya in 1 episode makanya lgsg sama dengan
                if new_scores > scores:
                    moves = 0 # reset move
                    scores = new_scores # get current scores

                total_reward += reward
                
                next_state = env.get_state()
                
                # Store the experience
                Agent.remember(state, current_action, reward, next_state, done)

                if step % 3 == 0: # Train every 3 steps
                    Agent.train_short_term(batch_size) # don't train only 1 row
                # this agent will remember current_state, action prediction in that state, and also the possible next state
                # makanya nnti kita bandingkan q_value sama next_possible_q_values, as in kita bandingkan kita punya hasil action prediction apakah bagus compare to next possible stage reward
                # kalau kau remember state setelah action, berarti kau bandingkan salah satu state diantara 3 possible states, hence the loss funciton is so low
                # if game over, just break the loop and end that game, as we want to train the short memory per game
                
                if done == 1: # meskipun game over, kita tetap mau remember itu state
                    state = env.reset()
                    break

                # Update the state
                state = next_state

            if moves > 0:
                loss = Agent.train_short_term(batch_size * 10) # train for each episode
                print(f"Epoch: {epoch+1}/{Agent.epochs}, episode: {episode+1}/{Agent.episode}, loss: {loss:.3f}, score: {scores}, reward: {total_reward:.1f}")
            else:
                print(f"Epoch: {epoch+1}/{Agent.epochs}, episode: {episode+1}/{Agent.episode}, no moves to train on, score: {scores}")
        
        if epoch % 5 == 0: # save every 5 epochs
            torch.save(model.state_dict(), "snake_model_q_learning.pth")


# Note:
# 1. kalau cmn 1D cnth [1] array dia itu size nya tertulis torch.Size([1]) -> size = () itu scalar (cmn value not in array)
# 2. kalau 2D, cnth [[1]] array dia tertulis (1,1)
# 3. buat torch.max(x, dim = n), basically compare max dimension n cnth kita punya (2,3,4), max di dim = 0 nnti jdi (3,4) bukan (1,3,4)
# 4. jangan lupa bikin [0] hbis torch max krn dia hsilnya in 2 element array (hsil max sama index letak max nya)
# 5. buat penjumlahan, need to make sense. either 1 + dimension apapun or sama dimension (imagine vector (2 element) + matrix (2 x 3)) -> doesn't make sense

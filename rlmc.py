import os
import time
from tqdm import trange
from collections import Counter

from models.causal_cnn import CausalCNNEncoder

from scipy.special import softmax
from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_data, get_model_info

DATA_DIR = 'dataset'
SCALE_MEAN, SCALE_STD = np.load(f'{DATA_DIR}/scaler.npy')
def inv_trans(x): return x * SCALE_STD + SCALE_MEAN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=100):
        super().__init__()
        self.cnn_encoder = CausalCNNEncoder(depth=3, kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=hidden_dim,
                                            reduced_size=hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.rank_embedding = nn.Embedding(act_dim, hidden_dim)
    
    def forward(self, obs, model_rank):
        ts_emb    = self.cnn_encoder(obs)
        model_emb = self.rank_embedding(model_rank).mean(axis=1)
        input_emb = F.relu(torch.cat([ts_emb, model_emb], 1))
        x = self.net(input_emb)
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=100):
        super().__init__()
        self.cnn_encoder = CausalCNNEncoder(depth=3, kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=hidden_dim,
                                            reduced_size=hidden_dim)
        self.act_layer = nn.Linear(act_dim, hidden_dim)
        self.fc_layer = nn.Linear(2*hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.rank_embedding = nn.Embedding(act_dim, hidden_dim)
    
    def forward(self, obs, act, model_rank):
        ts_emb    = self.cnn_encoder(obs)
        model_emb = self.rank_embedding(model_rank).mean(axis=1)
        input_emb = F.relu(torch.cat([ts_emb, model_emb], 1))
        x = F.relu(self.fc_layer(input_emb) + self.act_layer(act))
        x = self.out_layer(x)
        return x.squeeze()


class RLMCAgent:
    def __init__(self, states, ranks, obs_dim, act_dim, hidden_dim=100, lr=3e-4, gamma=0.99, tau=0.005):
        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        self.target_actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.target_critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.states = states
        self.ranks  = ranks

        self.gamma = gamma
        self.tau   = tau

        self.use_td = False

        # update the target network
        for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(param.data)
 
        def select_action(self, obs, model_rank):
            with torch.no_grad():
                action = self.actor(obs, model_rank).cpu().numpy()
            return softmax(action, axis=1)

    def update(self, sampled_obs_idxes, sampled_actions, sampled_rewards):
        batch_obs  = self.states[sampled_obs_idxes]  # (B, 7, 20)
        batch_rank = self.ranks[sampled_obs_idxes]   # (B, 4)
        with torch.no_grad():
            if self.use_td:
                batch_next_obs  = self.states[sampled_obs_idxes + 1]  # (B, 7, 30)
                batch_next_rank = self.ranks[sampled_obs_idxes + 1]
                target_q = self.target_critic(
                    batch_next_obs,
                    F.softmax(self.target_actor(batch_next_obs, batch_next_rank), dim=1),
                    batch_next_rank)  # (B,)
                target_q = sampled_rewards + self.gamma * target_q  # (B,)
            else:
                target_q = sampled_rewards

        current_q = self.critic(batch_obs, sampled_actions, batch_rank)
        q_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(batch_obs,
                                  F.softmax(self.actor(batch_obs, batch_rank), dim=1),
                                  batch_rank).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.use_td:
            for param, target_param in zip(
                    self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q_loss': q_loss.item(),
            'pi_loss': actor_loss.item(),
            'current_q': current_q.mean().item(),
            'target_q': target_q.mean().item()}

    def select_action(self, obs, rank):
        with torch.no_grad():
            action = self.actor(obs, rank).cpu().numpy()
        return softmax(action, axis=1)


# mape reward computed by the quantile
def get_mape_reward(q_mape, mape):
    q = 0
    while (q < 9) and (mape > q_mape[q]):
        q += 1
    reward = 1 - 2 * q / 9
    return reward


# mae reward computed by the quantile
def get_mae_reward(q_mae, mae):
    q = 0
    while (q < 9) and (mae > q_mae[q]):
        q += 1
    reward = 1 - 2 * q / 9
    return reward


# rank reward
def get_rank_reward(rank):
    reward = 1 - 2 * rank / 9
    return reward


#######
# Env #
#######
class Env:
    def __init__(self, train_error, train_y):
        self.error = train_error                                      # MAPE error df
        self.bm_preds = np.load(f'dataset/bm_train_preds.npy')     # (8448, 9, 24)
        self.y = train_y
    
    def reward_func(self, idx, action):
        if isinstance(action, int):  # one-hot action
            tmp = np.zeros(self.bm_preds.shape[1])
            tmp[action] = 1.
            action = tmp
        weighted_y = np.multiply(action.reshape(-1, 1), self.bm_preds[idx])     # (9, 24)
        weighted_y = weighted_y.sum(axis=0)                                     # (24,)  
        new_mape = mean_absolute_percentage_error(
            inv_trans(self.y[idx]), inv_trans(weighted_y))
        new_mae = mean_absolute_error(
            inv_trans(self.y[idx]), inv_trans(weighted_y))
        new_error = np.array([*self.error[idx], new_mape])
        rank = np.where(np.argsort(new_error) == len(new_error) - 1)[0][0]
        return rank, new_mape, new_mae 


class ReplayBuffer:
    def __init__(self, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # In TS data, `next_state` is just the S[i+1]
        self.states = np.zeros((max_size, 1), dtype=np.int32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        ind = np.random.randint(self.size, size=batch_size)
        states = self.states[ind].squeeze()
        actions = torch.FloatTensor(self.actions[ind]).to(device)
        rewards = torch.FloatTensor(self.rewards[ind]).to(device)
        return (states, actions, rewards.squeeze())


def sparse_explore(obs, act_dim):
    N = len(obs)
    x = np.zeros((N, act_dim))
    randn_idx = np.random.randint(0, act_dim, size=(N,))
    x[np.arange(N), randn_idx] = 1

    # disturb from the vertex
    delta = np.random.uniform(0.02, 0.1, size=(N, 1))
    x[np.arange(N), randn_idx] -= delta.squeeze()

    # noise
    noise = np.abs(np.random.randn(N, act_dim))
    noise[np.arange(N), randn_idx] = 0
    noise /= noise.sum(1, keepdims=True)
    noise = delta * noise
    sparse_action = x + noise

    return sparse_action


def random_explore(obs, act_dim):
    N = len(obs)
    random_action = np.random.normal(size=(N, act_dim))
    random_action = softmax(random_action, axis=1)
    return random_action


def evaluate_agent(agent, test_states, test_ranks, test_bm_preds, test_y):
    with torch.no_grad():
        weights = agent.select_action(test_states, test_ranks)  # (2816, 9)
    act_counter = Counter(weights.argmax(1))
    act_sorted  = sorted([(k, v) for k, v in act_counter.items()])
    weights = np.expand_dims(weights, -1)  # (2816, 9, 1)
    weighted_y = weights * test_bm_preds  # (2816, 9, 24)
    weighted_y = weighted_y.sum(1)  # (2816, 24)
    mae_loss = mean_absolute_error(inv_trans(test_y), inv_trans(weighted_y))
    mape_loss = mean_absolute_percentage_error(inv_trans(test_y), inv_trans(weighted_y))
    return mae_loss, mape_loss, act_sorted


def run_rlmc():
    (train_X, valid_X, test_X, train_y, valid_y, test_y, train_error, valid_error, test_error) = load_data()
    valid_preds = np.load(f'{DATA_DIR}/bm_valid_preds.npy')
    test_preds = np.load(f'{DATA_DIR}/bm_test_preds.npy')

    train_rank = get_model_info(train_error) 
    valid_rank = get_model_info(valid_error)
    test_rank  = get_model_info(test_error)
    train_rank = np.sort(train_rank[:, :4], axis=1)
    valid_rank = np.sort(valid_rank[:, :4], axis=1)
    test_rank  = np.sort(test_rank[:, :4], axis=1)

    # swap axes for CausalCNN input
    train_X = np.swapaxes(train_X, 2, 1)  # (N, 7, 96)
    valid_X = np.swapaxes(valid_X, 2, 1)  # (M, 7, 96)
    test_X  = np.swapaxes(test_X,  2, 1)  # (M, 7, 96)

    L = len(train_X) - 1
    FEAT_LEN = 20
    train_X = train_X[:, :, -FEAT_LEN:]  # (55928, 7, 10)
    valid_X = valid_X[:, :, -FEAT_LEN:]  # (6867,  7, 10)
    test_X  = test_X[:,  :, -FEAT_LEN:]  # (6867,  7, 10)

    # convert to torch.Tensor
    states = torch.FloatTensor(train_X).to(device)
    valid_states = torch.FloatTensor(valid_X).to(device)
    test_states = torch.FloatTensor(test_X).to(device)

    ranks = torch.LongTensor(train_rank).to(device)
    valid_ranks = torch.LongTensor(valid_rank).to(device)
    test_ranks = torch.LongTensor(test_rank).to(device)

    obs_dim = train_X.shape[1]  # 7
    act_dim = train_error.shape[-1]  # 9

    # environment that compute scores
    env = Env(train_error, train_y)

    if not os.path.exists('dataset/batch_buffer.csv'):
        batch_buffer = []
        for state_idx in trange(L, desc='[Create buffer]'):
            best_model_idx = train_error[state_idx].argmin()
            for action_idx in range(act_dim):
                rank, mape, mae = env.reward_func(state_idx, action_idx)
                batch_buffer.append((state_idx, action_idx, rank, mape, mae))
        batch_buffer_df = pd.DataFrame(
            batch_buffer,
            columns=['state_idx', 'action_idx', 'rank', 'mape', 'mae']) 
        batch_buffer_df.to_csv('dataset/batch_buffer.csv')
    else:
        batch_buffer_df = pd.read_csv('dataset/batch_buffer.csv', index_col=0)
    q_mape = [batch_buffer_df['mape'].quantile(0.1*i) for i in range(1, 10)]     
    q_mae = [batch_buffer_df['mape'].quantile(0.1*i) for i in range(1, 10)]     

    batch_buffer_df = batch_buffer_df.query(f'state_idx < {L}')

    # combined reward
    def get_batch_rewards(env, idxes, actions):
        rewards = []
        mae_lst = []
        for i in range(len(idxes)):
            rank, new_mape, new_mae = env.reward_func(idxes[i], actions[i])
            rank_reward = get_rank_reward(rank)
            mape_reward = get_mape_reward(q_mape, new_mape)
            # mae_reward  = get_mae_reward(q_mae, new_mae)
            combined_reward = mape_reward + rank_reward # + mae_reward
            mae_lst.append(new_mae)
            rewards.append(combined_reward)
        return rewards, mae_lst

    # run ddpg agent
    agent = RLMCAgent(states, ranks, obs_dim, act_dim)
    replay_buffer = ReplayBuffer(act_dim, max_size=int(5e5))
    extra_buffer = ReplayBuffer(act_dim, max_size=int(5e5))

    for _ in trange(10, desc='[Warm Up]'):
        shuffle_idxes   = np.random.randint(0, L, 300)
        sampled_states  = states[shuffle_idxes] 
        sampled_ranks   = ranks[shuffle_idxes]
        sampled_actions = agent.select_action(sampled_states, sampled_ranks)
        sampled_rewards, batch_mae = get_batch_rewards(env,
                                                       shuffle_idxes,
                                                       sampled_actions)
        for i in range(len(sampled_states)):
            replay_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])
            # if use_extra and batch_mae[i] >= q_mae[mae_threshold]:
            #     extra_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])

    step_size = 80
    step_num  = int(np.ceil(L / step_size))
    best_mape_loss = np.inf
    patience, max_patience = 0, 3
    for epoch in range(20):
        t1 = time.time()
        q_loss_lst, pi_loss_lst, q_lst, target_q_lst  = [], [], [], []
        shuffle_idx = np.random.permutation(np.arange(L))
        for i in range(step_num):
            batch_idx = shuffle_idx[i*step_size: (i+1)*step_size]        # (512,)
            batch_states = states[batch_idx]
            batch_ranks  = ranks[batch_idx]
            if np.random.random() < 0.8:
                # batch_actions = sparse_explore(batch_states, act_dim)
                batch_actions = random_explore(batch_states, act_dim)
            else:
                batch_actions = agent.select_action(batch_states, batch_ranks)
            batch_rewards, batch_mae = get_batch_rewards(env, batch_idx, batch_actions)
            for j in range(len(batch_idx)):
                replay_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])
                # if use_extra and batch_mae[j] >= q_mae[mae_threshold]:
                #     extra_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])

            # update ddpg
            sampled_obs_idxes, sampled_actions, sampled_rewards = replay_buffer.sample(512)
            info = agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards)

            pi_loss_lst.append(info['pi_loss'])
            q_loss_lst.append(info['q_loss'])
            q_lst.append(info['current_q'])
            target_q_lst.append(info['target_q'])

        valid_mae_loss, valid_mape_loss, count_lst = evaluate_agent(
            agent, valid_states, valid_ranks, valid_preds, valid_y)
        print(f'\n# Epoch {epoch + 1} ({(time.time() - t1)/60:.2f} min): '
              f'valid_mae_loss: {valid_mae_loss:.3f}\t'
              f'valid_mape_loss: {valid_mape_loss*100:.3f}\t' 
              f'q_loss: {np.average(q_loss_lst):.5f}\t'
              f'current_q: {np.average(q_lst):.5f}\t'
              f'target_q: {np.average(target_q_lst):.5f}\n'
              f'selected_act: {count_lst}\n')

        if valid_mape_loss < best_mape_loss:
            best_mape_loss = valid_mape_loss
            patience = 0
        else:
            patience += 1
        if patience == max_patience:
            break

    test_mae_loss, test_mape_loss, count_lst = evaluate_agent(
        agent, test_states, test_ranks, test_preds, test_y)
    print(f'test_mae_loss: {test_mae_loss:.3f}\t'
          f'test_mape_loss: {test_mape_loss*100:.3f}')
    return test_mae_loss, test_mape_loss



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--exp_name', default='rlmc', type=str)
    args = parser.parse_args()
    print(f'Exp args:\n{vars(args)}\n')

    seed         = args.seed
    exp_name     = args.exp_name

    np.random.seed(seed)
    torch.manual_seed(seed)

    mae_loss, mape_loss = run_rlmc()

    with open(f'{exp_name}_s{seed}.txt', 'w') as f:
        f.write(f'{mae_loss}, {mape_loss*100}\n')
 
from models.ddpg import Actor, Critic
from utils import load_data, evaluate_agent

import os
import time
import numpy as np
import pandas as pd
from tqdm import trange
from collections import Counter
from scipy.special import softmax
from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATA_DIR = 'dataset'
SCALE_MEAN, SCALE_STD = np.load(f'{DATA_DIR}/scaler.npy')
def inv_trans(x): return x * SCALE_STD + SCALE_MEAN


##############
# DDPG Agent #
##############
class DDPGAgent:
    def __init__(self, use_td, states, obs_dim, act_dim, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005):
        # initialize the actor & target_actor
        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        self.target_actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # initialize the critic
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.target_critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # training states
        self.states = states

        # parameters
        self.gamma  = gamma
        self.tau    = tau
        self.use_td = use_td

        # update the target network
        for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, obs):
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        return softmax(action, axis=1)

    def update(self,
               sampled_obs_idxes,
               sampled_actions,
               sampled_rewards,
               sampled_weights=None):
        batch_obs = self.states[sampled_obs_idxes]  # (512, 7, 20)

        with torch.no_grad():
            if self.use_td:
                # update w.r.t the TD target
                batch_next_obs = self.states[sampled_obs_idxes + 1]
                target_q = self.target_critic(
                    batch_next_obs, self.target_actor(batch_next_obs))  # (B,)
                target_q = sampled_rewards + self.gamma * target_q  # (B,)
            else:
                # without TD learning, just is supervised learning
                target_q = sampled_rewards
        current_q = self.critic(batch_obs, sampled_actions)     # (B,)

        # critic loss
        if sampled_weights is None:
            q_loss = F.mse_loss(current_q, target_q)
        else:
            # weighted mse loss
            q_loss = (sampled_weights * (current_q - target_q)**2).sum() /\
                sampled_weights.sum()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # actor loss ==> convert actor output to softmax weights
        if sampled_weights is None:
            actor_loss = -self.critic(
                batch_obs, F.softmax(self.actor(batch_obs), dim=1)).mean()
        else:
            # weighted actor loss
            actor_loss = -self.critic(batch_obs, F.softmax(self.actor(batch_obs), dim=1))
            actor_loss = (sampled_weights * actor_loss).sum() / sampled_weights.sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
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
            'target_q': target_q.mean().item()
        }


class Env:
    def __init__(self, train_error, train_y):
        self.error = train_error
        self.bm_preds = np.load(f'{DATA_DIR}/bm_train_preds.npy')
        self.y = train_y
    
    def reward_func(self, idx, action):
        if isinstance(action, int):
            tmp = np.zeros(self.bm_preds.shape[1])
            tmp[action] = 1.
            action = tmp
        weighted_y = np.multiply(action.reshape(-1, 1), self.bm_preds[idx])
        weighted_y = weighted_y.sum(axis=0)
        new_mape = mean_absolute_percentage_error(inv_trans(self.y[idx]), inv_trans(weighted_y))
        new_mae = mean_absolute_error(inv_trans(self.y[idx]), inv_trans(weighted_y))
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


def get_state_weight(train_error):
    L = len(train_error)
    best_model = train_error.argmin(1)
    best_model_counter = Counter(best_model)
    model_weight = {k:v/L for k,v in best_model_counter.items()}
    return model_weight


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


def pretrain_actor(obs_dim, act_dim, hidden_dim, states, train_error, cls_weights, 
                   valid_states, valid_error):
    best_train_model = torch.LongTensor(train_error.argmin(1)).to(device)
    best_valid_model = torch.LongTensor(valid_error.argmin(1)).to(device)

    actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
    best_actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
    cls_weights = torch.FloatTensor([1/cls_weights[w] for w in range(act_dim)]).to(device)

    L = len(states)
    batch_size = 512
    batch_num  = int(np.ceil(L / batch_size))
    optimizer  = torch.optim.Adam(actor.parameters(), lr=3e-4)
    loss_fn    = nn.CrossEntropyLoss(weight=cls_weights)  # weighted CE loss
    best_acc   = 0
    patience   = 0
    max_patience = 5
    for epoch in trange(200, desc='[Pretrain]'):
        epoch_loss = []
        shuffle_idx = np.random.permutation(np.arange(L))
        for i in range(batch_num):
            batch_idx = shuffle_idx[i*batch_size: (i+1)*batch_size]
            optimizer.zero_grad()
            batch_out = actor(states[batch_idx])
            loss = loss_fn(batch_out, best_train_model[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        with torch.no_grad():
            pred = actor(valid_states)
            pred_idx = pred.argmax(1)
            acc = (pred_idx == best_valid_model).sum() / len(pred)
        print(f'# epoch {epoch+1}: loss = {np.average(epoch_loss):.5f}\tacc = {acc:.3f}')

        # early stop w.r.t. validation acc
        if acc > best_acc:
            best_acc = acc
            patience = 0
            # update best model
            for param, target_param in zip(
                    actor.parameters(), best_actor.parameters()):
                target_param.data.copy_(param.data)
        else:
            patience += 1
        
        if patience == max_patience:
            break

    with torch.no_grad():
        pred = best_actor(valid_states)
        pred_idx = pred.argmax(1)
        acc = (pred_idx == best_valid_model).sum() / len(pred)    
    print(f'valid acc for pretrained actor: {acc:.3f}') 
    return best_actor


def run_rlmc(use_weight=True, use_td=True, use_extra=True, use_pretrain=True, epsilon=0.3):
    (train_X, valid_X, test_X, train_y, valid_y, test_y, train_error, valid_error, _) = load_data()
    valid_preds = np.load(f'{DATA_DIR}/bm_valid_preds.npy')
    test_preds = np.load(f'{DATA_DIR}/bm_test_preds.npy')
    train_X = np.swapaxes(train_X, 2, 1)
    valid_X = np.swapaxes(valid_X, 2, 1)
    test_X  = np.swapaxes(test_X,  2, 1)
    L = len(train_X) - 1 if use_td else len(train_X)
    FEAT_LEN = 20
    train_X = train_X[:, :, -FEAT_LEN:]
    valid_X = valid_X[:, :, -FEAT_LEN:]
    test_X  = test_X[:,  :, -FEAT_LEN:]
    states = torch.FloatTensor(train_X).to(device)
    valid_states = torch.FloatTensor(valid_X).to(device)
    test_states = torch.FloatTensor(test_X).to(device)

    obs_dim = train_X.shape[1]
    act_dim = train_error.shape[-1]

    env = Env(train_error, train_y)
    best_model_weight = get_state_weight(train_error)
    if not os.path.exists('dataset/batch_buffer.csv'):
        batch_buffer = []
        for state_idx in trange(L, desc='[Create buffer]'):
            best_model_idx = train_error[state_idx].argmin()
            for action_idx in range(act_dim):
                rank, mape, mae = env.reward_func(state_idx, action_idx)
                batch_buffer.append((state_idx, action_idx, rank, mape, mae, best_model_weight[best_model_idx]))
        batch_buffer_df = pd.DataFrame(
            batch_buffer,
            columns=['state_idx', 'action_idx', 'rank', 'mape', 'mae', 'weight']) 
        batch_buffer_df.to_csv('dataset/batch_buffer.csv')
    else:
        batch_buffer_df = pd.read_csv('dataset/batch_buffer.csv', index_col=0)
    q_mape = [batch_buffer_df['mape'].quantile(0.1*i) for i in range(1, 10)]     
    q_mae = [batch_buffer_df['mape'].quantile(0.1*i) for i in range(1, 10)]     

    if use_td:
        batch_buffer_df = batch_buffer_df.query(f'state_idx < {L}')

    def get_mape_reward(q_mape, mape, R=1):
        q = 0
        while (q < 9) and (mape > q_mape[q]):
            q += 1
        reward = -R + 2*R*(9 - q)/9
        return reward

    def get_mae_reward(q_mae, mae, R=1):
        q = 0
        while (q < 9) and (mae > q_mae[q]):
            q += 1
        reward = -R + 2*R*(9 - q)/9
        return reward
    
    def get_rank_reward(rank, R=1):
        reward = -R + 2*R*(9 - rank)/9
        return reward

    # combined reward
    def get_batch_rewards(env, idxes, actions):
        rewards = []
        mae_lst = []
        for i in range(len(idxes)):
            rank, new_mape, new_mae = env.reward_func(idxes[i], actions[i])
            rank_reward = get_rank_reward(rank, 1)
            mape_reward = get_mape_reward(q_mape, new_mape, 1)
            # mae_reward  = get_mae_reward(q_mae, new_mae, 2)
            combined_reward = mape_reward + rank_reward
            mae_lst.append(new_mae)
            rewards.append(combined_reward)
        return rewards, mae_lst

    # state weight
    state_weights = [1/best_model_weight[i] for i in train_error.argmin(1)]
    if use_weight:
        state_weights = torch.FloatTensor(state_weights).to(device)
    else:
        state_weights = None

    # initialize the DDPG agent
    agent = DDPGAgent(use_td, states, obs_dim, act_dim, hidden_dim=100, lr=1e-4)
    replay_buffer = ReplayBuffer(act_dim, max_size=int(1e5))
    extra_buffer  = ReplayBuffer(act_dim, max_size=int(1e5))
    if use_pretrain:
        pretrained_actor = pretrain_actor(obs_dim,
                                          act_dim,
                                          hidden_dim=100,
                                          states=states,
                                          train_error=train_error, 
                                          cls_weights=best_model_weight,
                                          valid_states=valid_states, 
                                          valid_error=valid_error)
        
        # copy the pretrianed actor 
        for param, target_param in zip(
                pretrained_actor.parameters(), agent.actor.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(
                pretrained_actor.parameters(), agent.target_actor.parameters()):
            target_param.data.copy_(param.data)

    # to save the best model
    best_actor = Actor(obs_dim, act_dim, hidden_dim=100).to(device)
    for param, target_param in zip(agent.actor.parameters(), best_actor.parameters()):
        target_param.data.copy_(param.data)

    # warm up
    for _ in trange(200, desc='[Warm Up]'):
        shuffle_idxes   = np.random.randint(0, L, 300)
        sampled_states  = states[shuffle_idxes] 
        sampled_actions = agent.select_action(sampled_states)
        sampled_rewards, _ = get_batch_rewards(env, shuffle_idxes, sampled_actions)
        for i in range(len(sampled_states)):
            replay_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])
            if use_extra and sampled_rewards[i] <= -1.:
                extra_buffer.add(shuffle_idxes[i], sampled_actions[i], sampled_rewards[i])

    step_size = 4
    step_num  = int(np.ceil(L / step_size))
    best_mape_loss = np.inf
    patience, max_patience = 0, 5
    for epoch in trange(500):
        t1 = time.time()
        q_loss_lst, pi_loss_lst, q_lst, target_q_lst  = [], [], [], []
        shuffle_idx = np.random.permutation(np.arange(L))
        for i in range(step_num):
            batch_idx = shuffle_idx[i*step_size: (i+1)*step_size]        # (512,)
            batch_states = states[batch_idx]
            if np.random.random() < epsilon:
                batch_actions = sparse_explore(batch_states, act_dim)
            else:
                batch_actions = agent.select_action(batch_states)
            batch_rewards, batch_mae = get_batch_rewards(env, batch_idx, batch_actions)
            for j in range(len(batch_idx)):
                replay_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])
                if use_extra and batch_rewards[j] <= -1.:
                    extra_buffer.add(batch_idx[j], batch_actions[j], batch_rewards[j])

            for _ in range(1):
                sampled_obs_idxes, sampled_actions, sampled_rewards = replay_buffer.sample(512)
                if use_weight:
                    sampled_weights = state_weights[sampled_obs_idxes]
                else:
                    sampled_weights = None
                info = agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards, sampled_weights)
                pi_loss_lst.append(info['pi_loss'])
                q_loss_lst.append(info['q_loss'])
                q_lst.append(info['current_q'])
                target_q_lst.append(info['target_q'])

                if use_extra and extra_buffer.ptr > 512:
                    sampled_obs_idxes, sampled_actions, sampled_rewards = extra_buffer.sample(512)
                    if use_weight:
                        sampled_weights = state_weights[sampled_obs_idxes]
                    else:
                        sampled_weights = None
                    info = agent.update(sampled_obs_idxes, sampled_actions, sampled_rewards, sampled_weights)
                    pi_loss_lst.append(info['pi_loss'])
                    q_loss_lst.append(info['q_loss'])
                    q_lst.append(info['current_q'])
                    target_q_lst.append(info['target_q'])

        valid_mae_loss, valid_mape_loss, count_lst = evaluate_agent(agent, valid_states, valid_preds, valid_y)
        print(f'\n# Epoch {epoch + 1} ({(time.time() - t1)/60:.2f} min): '
              f'valid_mae_loss: {valid_mae_loss:.3f}\t'
              f'valid_mape_loss: {valid_mape_loss*100:.3f}\t' 
              f'q_loss: {np.average(q_loss_lst):.5f}\t'
              f'current_q: {np.average(q_lst):.5f}\t'
              f'target_q: {np.average(target_q_lst):.5f}\n')

        if valid_mape_loss < best_mape_loss:
            best_mape_loss = valid_mape_loss
            patience = 0
            # save best model
            for param, target_param in zip(agent.actor.parameters(), best_actor.parameters()):
                target_param.data.copy_(param.data)
        else:
            patience += 1
        if patience == max_patience:
            break
        epsilon = max(epsilon-0.2, 0.1)
            
    for param, target_param in zip(agent.actor.parameters(), best_actor.parameters()):
        param.data.copy_(target_param)
    test_mae_loss, test_mape_loss, count_lst = evaluate_agent(
        agent, test_states, test_preds, test_y)
    print(f'test_mae_loss: {test_mae_loss:.3f}\t'
          f'test_mape_loss: {test_mape_loss*100:.3f}')

    return test_mae_loss, test_mape_loss


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use_weight',   action='store_true', default=False)
    parser.add_argument('--use_td',       action='store_false', default=True)
    parser.add_argument('--use_extra',    action='store_false', default=True)
    parser.add_argument('--use_pretrain', action='store_false', default=True)
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--exp_name', default='rlmc', type=str)
    args = parser.parse_args()
    print(f'Exp args:\n{vars(args)}\n')

    seed         = args.seed
    epsilon      = args.epsilon
    use_weight   = args.use_weight
    use_td       = args.use_td 
    use_extra    = args.use_extra
    use_pretrain = args.use_pretrain
    exp_name     = args.exp_name
    np.random.seed(seed)
    torch.manual_seed(seed)
    mae_loss, mape_loss = run_rlmc(use_weight, use_td, use_extra, use_pretrain,epsilon)

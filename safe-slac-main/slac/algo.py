import os

import numpy as np
import scipy.stats
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from slac.buffer import CostReplayBuffer, ReplayBuffer
from slac.network import GaussianPolicy, LatentModel, TwinnedQNetwork, SingleQNetwork
from slac.network.latent import CostLatentModel
from slac.network.sac import LatentGaussianPolicy
from slac.utils import create_feature_actions, grad_false, soft_update
from collections import defaultdict
import torch.nn.functional

class SlacAlgorithm:
    """
    Stochactic Latent Actor-Critic(SLAC).

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(
        self,
        state_shape,
        ometer_shape, 
        tgt_state_shape,
        action_shape,                         
        action_repeat,
        device,
        seed,
        gamma=0.99,
        batch_size_sac=256,
        batch_size_latent=32,
        buffer_size=10 ** 5,
        num_sequences=8,
        lr_sac=3e-4,
        lr_latent=1e-4,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
        tau=5e-3
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Replay buffer.
        self.buffer = ReplayBuffer(buffer_size, num_sequences, state_shape, ometer_shape, tgt_state_shape, action_shape, device)

        # Networks.
        self.actor = GaussianPolicy(action_shape, num_sequences, feature_dim, hidden_units).to(device)
        self.critic = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.critic_target = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.latent = LatentModel(state_shape, ometer_shape, tgt_state_shape, action_shape, feature_dim, z1_dim, z2_dim, hidden_units).to(device)
        soft_update(self.critic_target, self.critic, 1.0)
        grad_false(self.critic_target)

        # Target entropy is -|A|.
        self.target_entropy = -float(action_shape[0])
        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = torch.tensor([-10], requires_grad=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        # Optimizers.
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.state_shape = state_shape
        self.ometer_shape = ometer_shape
        self.tgt_state_shape = tgt_state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = gamma
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = tau

        # JIT compile to speed up.
        #fake_feature = torch.empty(1, num_sequences + 1, feature_dim, device=device)
        #fake_action = torch.empty(1, num_sequences, action_shape[0], device=device)
        #self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))
        self.create_feature_actions = create_feature_actions
        
        self.loss_image = torch.tensor(0.0)
        self.loss_actor = torch.tensor(0.0)
        self.loss_kld = torch.tensor(0.0)
        self.loss_reward = torch.tensor(0.0)
        self.loss_critic = torch.tensor(0.0)
        self.loss_alpha = torch.tensor(0.0) 
        self.entropy = torch.tensor(0.0)    

    def preprocess(self, ob):
        state = torch.tensor(ob.state, dtype=torch.uint8, device=self.device).float().div_(255.0)
        ometer = torch.tensor(ob.ometer, dtype=torch.float32, device=self.device).float().div_(150.0)
        with torch.no_grad():
            feature = self.latent.encoder(state, ometer).view(1, -1)
        action = torch.tensor(ob.action, dtype=torch.float32, device=self.device)
        feature_action = torch.cat([feature, action], dim=1)
        return feature_action

    def explore(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor.sample(feature_action)[0]
        return action.cpu().numpy()[0]

    def exploit(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(feature_action)
        return action.cpu().numpy()[0]

    def step(self, env, ob, t, is_pid):
        t += 1

        if is_pid:
            action = env.pid_sample()
        else:
            action = self.explore(ob)
        #env.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
        state, ometer, tgt_state, reward, done, _ = env.step(action)
        env.display()
        reward = reward
        #env.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
        mask = False if t == env._max_episode_steps else done
        ob.append(state, ometer, tgt_state, action)

        self.buffer.append(action, reward, mask, state, ometer, tgt_state, done)

        if done:
            t = 0
            state, ometer, tgt_state = env.reset()
            #env.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
            ob.reset_episode(state, ometer, tgt_state)
            #env.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
            self.buffer.reset_episode(state, ometer, tgt_state)

        return t

    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state_, ometer_, tgt_state_, action_, reward_, done_ = self.buffer.sample_latent(self.batch_size_latent)
        self.loss_kld, self.loss_image, self.loss_reward = self.latent.calculate_loss(state_, ometer_, tgt_state_, action_, reward_, done_)

        self.optim_latent.zero_grad()
        (self.loss_kld + self.loss_image + self.loss_reward).backward()
        self.optim_latent.step()


    def update_sac(self, writer):
        self.learning_steps_sac += 1
        state_, ometer_, tgt_state_, action_, reward, done = self.buffer.sample_sac(self.batch_size_sac)
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(state_, ometer_, action_)

        self.update_critic(z, next_z, action, next_feature_action, reward, done, writer)
        self.update_actor(z, feature_action, writer)
        soft_update(self.critic_target, self.critic, self.tau)

    def prepare_batch(self, state_, ometer_, action_):
        with torch.no_grad():
            # f(1:t+1)
            feature_ = self.latent.encoder(state_, ometer_)
            # z(1:t+1)
            z_ = torch.cat(self.latent.sample_posterior(feature_, action_)[2:4], dim=-1)

        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t)
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)

        return z, next_z, action, feature_action, next_feature_action

    def update_critic(self, z, next_z, action, next_feature_action, reward, done, writer):
        curr_q1, curr_q2 = self.critic(z, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_feature_action)
            next_q1, next_q2 = self.critic_target(next_z, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi
        target_q = reward + (1.0 - done) * self.gamma * next_q
        self.loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()

        self.optim_critic.zero_grad()
        self.loss_critic.backward(retain_graph=False)
        self.optim_critic.step()


    def update_actor(self, z, feature_action, writer):
        action, log_pi = self.actor.sample(feature_action)
        q1, q2 = self.critic(z, action)
        self.loss_actor = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi)

        self.optim_actor.zero_grad()
        self.loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        with torch.no_grad():
            self.entropy = -log_pi.detach().mean()
        self.loss_alpha = -self.log_alpha * (self.target_entropy - self.entropy)

        self.optim_alpha.zero_grad()
        self.loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We don't save target network to reduce workloads.
        torch.save(self.latent.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        torch.save(self.latent.decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
        torch.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(save_dir, "critic_target.pth"))
        torch.save(self.safety_critic.state_dict(), os.path.join(save_dir, "safety_critic.pth"))
        torch.save(self.safety_critic_target.state_dict(), os.path.join(save_dir, "safety_critic_target.pth"))
        
    def load_model(self, load_dir):
        if os.path.exists(os.path.join(load_dir, "encoder.pth")):
            self.latent.encoder.load_state_dict(torch.load(os.path.join(load_dir, "encoder.pth")))
        if os.path.exists(os.path.join(load_dir, "decoder.pth")):
            self.latent.decoder.load_state_dict(torch.load(os.path.join(load_dir, "decoder.pth")))
        if os.path.exists(os.path.join(load_dir, "latent.pth")):
            self.latent.load_state_dict(torch.load(os.path.join(load_dir, "latent.pth")))
        if os.path.exists(os.path.join(load_dir, "actor.pth")):
            self.actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pth")))
        if os.path.exists(os.path.join(load_dir, "critic.pth")):
            self.critic.load_state_dict(torch.load(os.path.join(load_dir, "critic.pth")))
        if os.path.exists(os.path.join(load_dir, "critic_target.pth")):
            self.critic_target.load_state_dict(torch.load(os.path.join(load_dir, "critic_target.pth")))
        if os.path.exists(os.path.join(load_dir, "safety_critic.pth")):
            self.safety_critic.load_state_dict(torch.load(os.path.join(load_dir, "safety_critic.pth")))
        if os.path.exists(os.path.join(load_dir, "safety_critic_target.pth")):
            self.safety_critic_target.load_state_dict(torch.load(os.path.join(load_dir, "safety_critic_target.pth")))

class SafetyCriticSlacAlgorithm:
    """
    History-based safe SLAC algorithm.
    """

    def __init__(
        self,
        state_shape,
        ometer_shape,
        tgt_state_shape,
        action_shape,                         
        action_repeat,
        device,
        seed,
        gamma=0.99,
        gamma_c=0.995,
        batch_size_sac=256,
        batch_size_latent=32,
        buffer_size=10 ** 5,
        num_sequences=8,
        lr_sac=3e-4,
        lr_latent=1e-4,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
        tau=5e-3,
        start_alpha=3.3e-4,
        start_lagrange=2.5e-2,
        grad_clip_norm=10.0,
        image_noise=0.1
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        budget = 100
        self.budget_undiscounted = budget
        self.steps = 1000/action_repeat
        self.budget = budget*(1 - gamma_c ** (1000/action_repeat)) / (1 - gamma_c)/(1000/action_repeat)

        # Replay buffer.
        self.buffer = CostReplayBuffer(buffer_size, num_sequences, state_shape, ometer_shape, tgt_state_shape, action_shape, device)
        self.grad_clip_norm = grad_clip_norm
        # Networks.
        
        self.actor = GaussianPolicy(action_shape, num_sequences, feature_dim, hidden_units)
        self.critic = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units)
        self.critic_target = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units)
        self.safety_critic = SingleQNetwork(action_shape, z1_dim, z2_dim, hidden_units, init_output=self.budget)
        self.safety_critic_target = SingleQNetwork(action_shape, z1_dim, z2_dim, hidden_units, init_output=self.budget)
        self.latent = CostLatentModel(state_shape, ometer_shape, tgt_state_shape, action_shape, feature_dim, z1_dim, z2_dim, hidden_units, image_noise=image_noise)
        soft_update(self.critic_target, self.critic, 1.0)
        soft_update(self.safety_critic_target, self.safety_critic, 1.0)
        
        parts = [(self.actor, None, "actor"),
        (self.critic, None, "critic"),
        (self.critic_target, None, "critic_target"),
        (self.safety_critic, None, "safety_critic"),
        (self.safety_critic_target, None, "safety_critic_target"),
        (self.latent, None, "latent")]
        for model, optimizer, name in parts:
            model.to(device)
            if "target" not in name:
                model.train()

        grad_false(self.critic_target)
        grad_false(self.safety_critic_target)

        # Target entropy is -|A|.
        self.target_entropy = -np.prod(action_shape)
        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = torch.tensor([np.log(start_alpha)], requires_grad=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()
        
        self.raw_lag = torch.tensor([np.log(np.exp(start_lagrange)-1)], requires_grad=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.lagrange = torch.nn.functional.softplus(self.raw_lag)


        # Optimizers.
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_safety_critic = Adam(self.safety_critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)
        self.optim_lagrange = SGD([self.raw_lag], lr=2e-4)
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

        self.sched_actor = MultiStepLR(self.optim_actor, milestones=[400], gamma=0.5)
        self.sched_critic = MultiStepLR(self.optim_critic, milestones=[400], gamma=0.5)
        self.sched_safety_critic = MultiStepLR(self.optim_safety_critic, milestones=[400], gamma=0.5)
        self.sched_alpha = MultiStepLR(self.optim_alpha, milestones=[400], gamma=0.5)
        self.sched_lagrange = MultiStepLR(self.optim_lagrange, milestones=[400], gamma=0.5)
        self.sched_latent = MultiStepLR(self.optim_latent, milestones=[400], gamma=0.5)
        self.scheds = [self.sched_actor, self.sched_critic, self.sched_safety_critic, self.sched_alpha, self.sched_lagrange, self.sched_latent]
        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.state_shape = state_shape
        self.ometer_shape = ometer_shape
        self.tgt_state_shape = tgt_state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = gamma
        self.gamma_c = gamma_c
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = tau

        self.epoch_len = 30_000//self.action_repeat
        self.epoch_costreturns = []
        self.epoch_rewardreturns = []
        self.episode_costreturn = 0
        self.episode_rewardreturn = 0
        
        self.loss_averages = defaultdict(lambda : 0)

        # JIT compile to speed up.
        #fake_feature = torch.empty(1, num_sequences + 1, feature_dim, device=device)
        #fake_action = torch.empty(1, num_sequences, action_shape[0], device=device)
        #self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))
        self.create_feature_actions = create_feature_actions

        self.loss_image = torch.tensor(0.0)
        self.loss_actor = torch.tensor(0.0)
        self.loss_kld = torch.tensor(0.0)
        self.loss_reward = torch.tensor(0.0)
        self.loss_critic = torch.tensor(0.0)
        self.loss_safety_critic = torch.tensor(0.0)
        self.loss_cost = torch.tensor(0.0)
        self.loss_alpha = torch.tensor(0.0) 
        self.loss_lag = torch.tensor(0.0) 
        self.entropy = torch.tensor(0.0) 

    def preprocess(self, ob):
        state = torch.tensor(ob.state, dtype=torch.uint8, device=self.device).float().div_(255.0)
        ometer = torch.tensor(ob.ometer, dtype=torch.float32, device=self.device).float().div_(150.0)
        with torch.no_grad():
            feature = self.latent.encoder(state, ometer).view(1, -1)
        action = torch.tensor(ob.action, dtype=torch.float32, device=self.device)
        feature_action = torch.cat([feature, action], dim=1)
        return feature_action

    def explore(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor.sample(feature_action)[0]
        return action.cpu().numpy()[0]

    def exploit(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(feature_action)
        return action.cpu().numpy()[0]

    def step(self, env, ob, t, is_pid, writer=None):
        
        t += 1

        if is_pid:
            action = action = env.pid_sample()
        else:
            action = self.explore(ob)
        #env.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
        state, ometer, tgt_state, reward, done, info = env.step(action)
        env.display()
        cost = info["cost"]
        self.lastcost = cost
        self.episode_costreturn += cost
        self.episode_rewardreturn += reward
        mask = False if t >= env._max_episode_steps else done
        ob.append(state, ometer, tgt_state, action)

        self.buffer.append(action, reward, mask, state, ometer, tgt_state, done, cost)

        if done:
            if not is_pid:
                self.last_costreturn = self.episode_costreturn
                
                self.epoch_costreturns.append(self.episode_costreturn)
                self.epoch_rewardreturns.append(self.episode_rewardreturn)
            self.episode_costreturn = 0
            self.episode_rewardreturn = 0
            t = 0
            state, ometer, tgt_state = env.reset()
            ob.reset_episode(state, ometer, tgt_state)
            self.buffer.reset_episode(state, ometer, tgt_state)
        return t

    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state_, ometer_, tgt_state_, action_, reward_, done_, cost_ = self.buffer.sample_latent(self.batch_size_latent)
        self.loss_kld, self.loss_image, self.loss_reward, self.loss_cost = self.latent.calculate_loss(state_, ometer_, tgt_state_, action_, reward_, done_, cost_)

        self.optim_latent.zero_grad()
        (self.loss_kld + self.loss_image + self.loss_reward + self.loss_cost).backward()
        torch.nn.utils.clip_grad_norm_(self.latent.parameters(), self.grad_clip_norm)
        self.optim_latent.step()


    def update_sac(self, writer):
        self.learning_steps_sac += 1
        state_, ometer_, tgt_state_, action_, reward, done, cost = self.buffer.sample_sac(self.batch_size_sac)
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(state_, ometer_, action_)

        self.update_critic(z, next_z, action, next_feature_action, reward, done, writer)
        self.update_safety_critic(z, next_z, action, next_feature_action, cost, done, writer)
        self.update_actor(z, feature_action, writer)
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.safety_critic_target, self.safety_critic, self.tau)

    def prepare_batch(self, state_, ometer_, action_):
        with torch.no_grad():
            # f(1:t+1)
            feature_ = self.latent.encoder(state_, ometer_)
            # z(1:t+1)
            z_ = torch.cat(self.latent.sample_posterior(feature_, action_)[2:4], dim=-1)

        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t)
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)

        return z, next_z, action, feature_action, next_feature_action

    def update_critic(self, z, next_z, action, next_feature_action, reward, done, writer):
        curr_q1, curr_q2 = self.critic(z, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_feature_action)
            next_q1, next_q2 = self.critic_target(next_z, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi
        target_q = reward + (1.0 - done) * self.gamma * next_q
        self.loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()

        self.optim_critic.zero_grad()
        self.loss_critic.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.optim_critic.step()


    def update_safety_critic(self, z, next_z, action, next_feature_action, cost, done, writer):
        curr_c1 = self.safety_critic(z, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_feature_action)
            next_c = self.safety_critic_target(next_z, next_action)
            target_c = cost + (1.0 - done) * self.gamma_c * next_c
        self.loss_safety_critic = torch.nn.functional.mse_loss(curr_c1, target_c)

        self.optim_safety_critic.zero_grad()
        self.loss_safety_critic.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.safety_critic.parameters(), self.grad_clip_norm)
        self.optim_safety_critic.step()

    def update_actor(self, z, feature_action, writer):
        action, log_pi = self.actor.sample(feature_action)
        q1, q2 = self.critic(z, action)
        c1 = self.safety_critic(z, action)
        # Use damping
        with torch.no_grad():
            budget_diff = (self.budget-c1)
            budget_remainder = budget_diff.mean()

        self.loss_actor = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi - self.lagrange.detach() * c1, dim=0)

        self.optim_actor.zero_grad()
        self.loss_actor.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm*2)
        self.optim_actor.step()

        with torch.no_grad():
            self.entropy = -log_pi.detach().mean()   
        self.loss_alpha = -self.log_alpha * (self.target_entropy - self.entropy)

        self.optim_alpha.zero_grad()
        self.loss_alpha.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.log_alpha, self.grad_clip_norm)
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()


    def update_lag(self, t, writer):
        try:
            last_cost = self.lastcost
        except:
            return
        self.loss_lag = (torch.nn.functional.softplus(self.raw_lag)/torch.nn.functional.softplus(self.raw_lag).detach() * (self.budget_undiscounted/self.steps-last_cost)).mean()


        self.optim_lagrange.zero_grad()
        self.loss_lag.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.raw_lag, self.grad_clip_norm*50)

        self.optim_lagrange.step()
        with torch.no_grad():
            self.lagrange = torch.nn.functional.softplus(self.raw_lag)

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We don't save target network to reduce workloads.
        torch.save(self.latent.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        torch.save(self.latent.decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
        torch.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(save_dir, "critic_target.pth"))
        torch.save(self.safety_critic.state_dict(), os.path.join(save_dir, "safety_critic.pth"))
        torch.save(self.safety_critic_target.state_dict(), os.path.join(save_dir, "safety_critic_target.pth"))
        
    def load_model(self, load_dir):
        if os.path.exists(os.path.join(load_dir, "encoder.pth")):
            self.latent.encoder.load_state_dict(torch.load(os.path.join(load_dir, "encoder.pth")))
        if os.path.exists(os.path.join(load_dir, "decoder.pth")):
            self.latent.decoder.load_state_dict(torch.load(os.path.join(load_dir, "decoder.pth")))
        if os.path.exists(os.path.join(load_dir, "latent.pth")):
            self.latent.load_state_dict(torch.load(os.path.join(load_dir, "latent.pth")))
        if os.path.exists(os.path.join(load_dir, "actor.pth")):
            self.actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pth")))
        if os.path.exists(os.path.join(load_dir, "critic.pth")):
            self.critic.load_state_dict(torch.load(os.path.join(load_dir, "critic.pth")))
        if os.path.exists(os.path.join(load_dir, "critic_target.pth")):
            self.critic_target.load_state_dict(torch.load(os.path.join(load_dir, "critic_target.pth")))
        if os.path.exists(os.path.join(load_dir, "safety_critic.pth")):
            self.safety_critic.load_state_dict(torch.load(os.path.join(load_dir, "safety_critic.pth")))
        if os.path.exists(os.path.join(load_dir, "safety_critic_target.pth")):
            self.safety_critic_target.load_state_dict(torch.load(os.path.join(load_dir, "safety_critic_target.pth")))

class LatentPolicySafetyCriticSlac(SafetyCriticSlacAlgorithm):
    """
    Latent state-based safe SLAC algorithm.
    """

    def __init__(
        self,
        state_shape,
        ometer_shape,
        tgt_state_shape,
        action_shape,                         
        action_repeat,
        device,
        seed,
        gamma=0.99,
        gamma_c=0.995,
        batch_size_sac=256,
        batch_size_latent=32,
        buffer_size=10 ** 5,
        num_sequences=8,
        lr_sac=3e-4,
        lr_latent=1e-4,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
        tau=5e-3,
        start_alpha=3.3e-4,
        start_lagrange=2.5e-2,
        grad_clip_norm=10.0,
        image_noise=0.1
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        budget = 25
        self.budget_undiscounted = budget
        self.steps = 1000/action_repeat
        self.budget = budget*(1 - gamma_c ** (1000/action_repeat)) / (1 - gamma_c)/(1000/action_repeat)

        # Replay buffer.
        self.buffer = CostReplayBuffer(buffer_size, num_sequences, state_shape, ometer_shape, tgt_state_shape, action_shape, device)
        self.grad_clip_norm = grad_clip_norm
        # Networks.
        
        self.actor = LatentGaussianPolicy(action_shape, z1_dim, z2_dim, hidden_units)
        self.critic = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units)
        self.critic_target = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units)
        self.safety_critic = SingleQNetwork(action_shape, z1_dim, z2_dim, hidden_units, init_output=self.budget)
        self.safety_critic_target = SingleQNetwork(action_shape, z1_dim, z2_dim, hidden_units, init_output=self.budget)
        self.latent = CostLatentModel(state_shape, ometer_shape, tgt_state_shape, action_shape, feature_dim, z1_dim, z2_dim, hidden_units, image_noise=image_noise)
        soft_update(self.critic_target, self.critic, 1.0)
        soft_update(self.safety_critic_target, self.safety_critic, 1.0)
        
        parts = [(self.actor, None, "actor"),
        (self.critic, None, "critic"),
        (self.critic_target, None, "critic_target"),
        (self.safety_critic, None, "safety_critic"),
        (self.safety_critic_target, None, "safety_critic_target"),
        (self.latent, None, "latent")]
        for model, optimizer, name in parts:
            model.to(device)
            if "target" not in name:
                model.train()

        grad_false(self.critic_target)
        grad_false(self.safety_critic_target)

        # Target entropy is -|A|.
        self.target_entropy = -np.prod(action_shape)*1.0
        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = torch.tensor([np.log(start_alpha)], requires_grad=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()
        
        self.raw_lag = torch.tensor([np.log(np.exp(start_lagrange)-1)], requires_grad=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.lagrange = torch.nn.functional.softplus(self.raw_lag)
        # Optimizers.
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_safety_critic = Adam(self.safety_critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)
        self.optim_lagrange = SGD([self.raw_lag], lr=2e-4)
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

        self.sched_actor = MultiStepLR(self.optim_actor, milestones=[400], gamma=0.5)
        self.sched_critic = MultiStepLR(self.optim_critic, milestones=[400], gamma=0.5)
        self.sched_safety_critic = MultiStepLR(self.optim_safety_critic, milestones=[400], gamma=0.5)
        self.sched_alpha = MultiStepLR(self.optim_alpha, milestones=[400], gamma=0.5)
        self.sched_lagrange = MultiStepLR(self.optim_lagrange, milestones=[400], gamma=0.5)
        self.sched_latent = MultiStepLR(self.optim_latent, milestones=[400], gamma=0.5)
        self.scheds = [self.sched_actor, self.sched_critic, self.sched_safety_critic, self.sched_alpha, self.sched_lagrange, self.sched_latent]

        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.state_shape = state_shape
        self.ometer_shape = ometer_shape
        self.tgt_state_shape = tgt_state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = gamma
        self.gamma_c = gamma_c
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = tau

        self.epoch_len = 30_000//self.action_repeat
        self.epoch_costreturns = []
        self.epoch_rewardreturns = []
        self.episode_costreturn = 0
        self.episode_rewardreturn = 0
        
        self.loss_averages = defaultdict(lambda : 0)

        # JIT compile to speed up.
        #fake_feature = torch.empty(1, num_sequences + 1, feature_dim, device=device)
        #fake_action = torch.empty(1, num_sequences, action_shape[0], device=device)
        #self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))
        self.z1 = None
        self.z2 = None
        self.create_feature_actions = create_feature_actions

    def step(self, env, ob, t, is_pid, writer=None):
        
        t += 1

        if is_pid:
            action = env.pid_sample()
        else:
            action = self.explore(ob)
        #env.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
        state, ometer, tgt_state, reward, done, info = env.step(action)
        env.display()
        cost = info["cost"]
        self.lastcost = cost
        self.episode_costreturn += cost
        self.episode_rewardreturn += reward
        mask = False if t >= env._max_episode_steps else done
        ob.append(state, ometer, tgt_state, action)

        self.buffer.append(action, reward, mask, state, ometer, tgt_state, done, cost)

        if done:
            if not is_pid:
                self.last_costreturn = self.episode_costreturn
                
                self.epoch_costreturns.append(self.episode_costreturn)
                self.epoch_rewardreturns.append(self.episode_rewardreturn)
            self.episode_costreturn = 0
            self.episode_rewardreturn = 0
            t = 0
            state, ometer, tgt_state = env.reset()
            ob.reset_episode(state, ometer, tgt_state)
            self.buffer.reset_episode(state, ometer, tgt_state)
            self.z1 = None
            self.z2 = None
        return t

    def preprocess(self, ob):
        state = torch.tensor(ob.last_state, dtype=torch.uint8, device=self.device).float().div_(255.0)
        ometer = torch.tensor(ob.last_ometer, dtype=torch.float32, device=self.device).float().div_(150.0)
        with torch.no_grad():
            feature = self.latent.encoder(state.unsqueeze(0),ometer.unsqueeze(0))
        action = torch.tensor(ob.last_action, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
            
        if self.z2 is None:
            z1_mean, z1_std = self.latent.z1_posterior_init(feature)
            self.z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            z2_mean, z2_std = self.latent.z2_posterior_init(self.z1)
            self.z2 = z2_mean + torch.randn_like(z2_std) * z2_std
        else:
            z1_mean, z1_std = self.latent.z1_posterior(torch.cat([feature, self.z2, action], dim=-1))
            self.z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.latent.z2_posterior(torch.cat([self.z1, self.z2, action], dim=-1))
            self.z2 = z2_mean + torch.randn_like(z2_std) * z2_std
        
        return torch.cat([self.z1,self.z2],dim=-1)

    def explore(self, ob):
        z = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor.sample(z)[0][0]
        return action.cpu().numpy()[0]

    def exploit(self, ob):
        z = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(z)
        return action.cpu().numpy()[0][0]

    def update_actor(self, z, feature_action, writer):
        action, log_pi = self.actor.sample(z)
        q1, q2 = self.critic(z, action)
        c1 = self.safety_critic(z, action)
        
        with torch.no_grad():
            budget_diff = (self.budget-c1)
            budget_remainder = budget_diff.mean()

        self.loss_actor = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi - (self.lagrange.detach()) * c1, dim=0)

        self.optim_actor.zero_grad()
        self.loss_actor.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm*2)
        self.optim_actor.step()

        with torch.no_grad():
            self.entropy = -log_pi.detach().mean()   
        self.loss_alpha = -self.log_alpha * (self.target_entropy - self.entropy)

        self.optim_alpha.zero_grad()
        self.loss_alpha.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.log_alpha, self.grad_clip_norm)
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()


    def update_lag(self, t, writer):
        try:
            last_cost = self.lastcost
        except:
            return
        self.loss_lag = (torch.nn.functional.softplus(self.raw_lag)/torch.nn.functional.softplus(self.raw_lag).detach() * (self.budget_undiscounted/self.steps-last_cost)).mean()


        self.optim_lagrange.zero_grad()
        self.loss_lag.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.raw_lag, self.grad_clip_norm*50)

        self.optim_lagrange.step()
        with torch.no_grad():
            self.lagrange = torch.nn.functional.softplus(self.raw_lag)

    def update_sac(self, writer):
        self.learning_steps_sac += 1
        state_, ometer_, tgt_state_, action_, reward, done, cost = self.buffer.sample_sac(self.batch_size_sac)
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(state_, ometer_, action_)


        self.update_critic(z, next_z, action, next_feature_action, reward, done, writer)
        self.update_safety_critic(z, next_z, action, next_feature_action, cost, done, writer)
        self.update_actor(z, feature_action, writer)
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.safety_critic_target, self.safety_critic, self.tau)

    def update_critic(self, z, next_z, action, next_feature_action, reward, done, writer):
        curr_q1, curr_q2 = self.critic(z, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_z)
            next_q1, next_q2 = self.critic_target(next_z, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi
        target_q = reward + (1.0 - done) * self.gamma * next_q
        self.loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()

        self.optim_critic.zero_grad()
        self.loss_critic.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.optim_critic.step()

    def update_safety_critic(self, z, next_z, action, next_feature_action, cost, done, writer):
        curr_c1 = self.safety_critic(z, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_z)
            next_c = self.safety_critic_target(next_z, next_action)
            target_c = cost + (1.0 - done) * self.gamma_c * next_c
        self.loss_safety_critic = torch.nn.functional.mse_loss(curr_c1, target_c)

        self.optim_safety_critic.zero_grad()
        self.loss_safety_critic.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.safety_critic.parameters(), self.grad_clip_norm)
        self.optim_safety_critic.step()
    
    # prior sample learning    
    def update_sac_with_prior_sample(self, writer):
        state_, ometer_, _, _, _, _, _ = self.buffer.sample_sac(self.batch_size_sac)
        
        with torch.no_grad():
            ## p(z1(0)) = N(0, I)
            #z1_mean, z1_std = self.latent.z1_prior_init(torch.zeros([self.batch_size_sac,self.action_shape[0]]).to(self.device))
            # f(1:t+1)
            feature_ = self.latent.encoder(state_, ometer_)
            # q(z1(0) | feat(0))
            z1_mean, z1_std = self.latent.z1_posterior_init(feature_[:, 0])
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # p(z2(0) | z1(0))
            z2_mean, z2_std = self.latent.z2_prior_init(z1)
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std
            z = torch.cat([z1,z2],dim=-1)
        
            action = self.actor.sample(z)[0]
        
            # p(z1(t) | z2(t-1), a(t-1))
            next_z1_mean, next_z1_std = self.latent.z1_prior(torch.cat([z2, action],dim=-1))
            next_z1 = next_z1_mean + torch.randn_like(next_z1_std) * next_z1_std
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            next_z2_mean, next_z2_std = self.latent.z2_prior(torch.cat([next_z1, z2, action], dim=-1))
            next_z2 = next_z2_mean + torch.randn_like(next_z2_std) * next_z2_std
            next_z = torch.cat([next_z1,next_z2],dim=-1)
        
            reward_mean, reward_std = self.latent.reward(torch.cat([z1,z2,action,next_z1,next_z2],dim=-1))
            reward = reward_mean + torch.randn_like(reward_std) * reward_std
            cost_mean, cost_std = self.latent.cost(torch.cat([z1,z2,action,next_z1,next_z2],dim=-1))
            cost = cost_mean + torch.randn_like(cost_std) * cost_std
        
        
        self.update_critic(z, next_z, action, None, reward, False, writer)
        self.update_safety_critic(z, next_z, action, None, cost, False, writer)
        self.update_actor(z, None, writer)
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.safety_critic_target, self.safety_critic, self.tau)


import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from slac.network.initializer import initialize_weight
#from slac.network.ITTR_pytorch import HPB
#from slac.network.vit_for_small_dataset import ViT
from slac.utils import build_mlp, calculate_kl_divergence


class FixedGaussian(nn.Module):
    """
    Fixed diagonal gaussian distribution.
    """

    def __init__(self, output_dim, std):
        super(FixedGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std


class Gaussian(nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(Gaussian, self).__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.ELU(),
        ).apply(initialize_weight)

    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        return mean, std

class Bernoulli(nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(Bernoulli, self).__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.ELU(),
        ).apply(initialize_weight)

    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        p = torch.sigmoid(x)
        return p


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, input_dim=288, output_dim=3, std=1.0):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, output_dim, 5, 2, 2, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        ).apply(initialize_weight)
        """
        self.net = ReCon(image_size=64, 
            patch_size=16, 
            dim=input_dim, 
            depth=1, 
            heads=8, 
            mlp_dim=input_dim*4, 
            channels = 3, 
            dim_head = 64, 
            dropout = 0., 
            emb_dropout = 0.).apply(initialize_weight)
        """
        self.std = std

    def forward(self, x):
        B, S, latent_dim = x.size()
        x = x.view(B * S, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(B, S, C, W, H)
        return x, torch.ones_like(x).mul_(self.std)

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, input_dim=6, ometer_dim=80, output_dim=256):
        super(Encoder, self).__init__()
        """
        self.net = ViT(image_size = 64,
            patch_size = 16,
            dim = 512,
            depth = 4,
            heads = 8,
            mlp_dim = 1024,
            output_dim = output_dim,
            channels = input_dim, 
            dim_head = 64,
            dropout = 0.1,
            emb_dropout = 0.1).apply(initialize_weight)
        """
        self.net = nn.Sequential(
            # (6, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.ELU(inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ELU(inplace=True),
            # (64, 16, 16) -> (64, 16, 16)
            # HPB(64,16,4,2,8,8,0.2,0.2),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ELU(inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ELU(inplace=True),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.ELU(inplace=True),
        ).apply(initialize_weight)

        self.cat_net = nn.Sequential(
            nn.Linear(output_dim + ometer_dim, output_dim),
            nn.ELU(inplace=True),
        ).apply(initialize_weight)

    def forward(self, x, y):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W)
        x = self.net(x)
        x = x.view(B, S, -1)
        y = y.view(B, S, -1)
        y = torch.cat((x,y),dim=-1)
        y = self.cat_net(y)
        return y


class LatentModel(nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics and the reward.
    """

    def __init__(
        self,
        state_shape,
        ometer_shape,
        tgt_state_shape,
        action_shape,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
    ):
        super(LatentModel, self).__init__()
        # p(z1(0)) = N(0, I)
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = Gaussian(z1_dim, z2_dim, hidden_units)
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = Gaussian(
            z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = Gaussian(
            z1_dim + z2_dim + action_shape[0],
            z2_dim,
            hidden_units,
        )

        # q(z1(0) | feat(0))
        self.z1_posterior_init = Gaussian(feature_dim, z1_dim, hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(
            feature_dim + z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = Gaussian(
            2 * z1_dim + 2 * z2_dim + action_shape[0],
            1,
            hidden_units,
        )
        

        # feat(t) = Encoder(x(t))
        self.encoder = Encoder(state_shape[0], ometer_shape[0]*ometer_shape[1],feature_dim)
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            z1_dim + z2_dim,
            tgt_state_shape[0],
            std=np.sqrt(0.04),
        )
        self.apply(initialize_weight)

    def sample_prior(self, actions_, z2_post_):
        # p(z1(0)) = N(0, I)
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # p(z1(t) | z2(t-1), a(t-1))
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        # Concatenate initial and consecutive latent variables
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return (z1_mean_, z1_std_)

    def sample_posterior(self, features_, actions_):
        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, actions_.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return (z1_mean_, z1_std_, z1_, z2_)

    def calculate_loss(self, state_, ometer_, tgt_state_, action_, reward_, done_):
        # Calculate the sequence of features.
        feature_ = self.encoder(state_, ometer_)

        # Sample from latent variable model.
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # Prediction loss of images.
        z_ = torch.cat([z1_, z2_], dim=-1)
        state_mean_, state_std_ = self.decoder(z_)
        state_noise_ = (tgt_state_ - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_tgt_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_image_tgt_ = -log_likelihood_tgt_.mean(dim=0).sum()
        loss_image = loss_image_tgt_

        # Prediction loss of rewards.
        x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean_, reward_std_ = self.reward(x.view(B * S, X))
        reward_mean_ = reward_mean_.view(B, S, 1)
        reward_std_ = reward_std_.view(B, S, 1)
        reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
        log_likelihood_reward_ = (-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()
        return loss_kld, loss_image, loss_reward

class CostLatentModel(nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics, reward and cost.
    """

    def __init__(
        self,
        state_shape,
        ometer_shape,
        tgt_state_shape,
        action_shape,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
        image_noise=0.1
    ):
        super(CostLatentModel, self).__init__()
        self.bceloss = torch.nn.BCELoss(reduction="none")
        # p(z1(0)) = N(0, I)
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = Gaussian(z1_dim, z2_dim, hidden_units)
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = Gaussian(
            z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = Gaussian(
            z1_dim + z2_dim + action_shape[0],
            z2_dim,
            hidden_units,
        )

        # q(z1(0) | feat(0))
        self.z1_posterior_init = Gaussian(feature_dim, z1_dim, hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(
            feature_dim + z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = Gaussian(
            2 * z1_dim + 2 * z2_dim + action_shape[0],
            1,
            hidden_units,
        )

        # p(c(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.cost = Gaussian(
            2 * z1_dim + 2 * z2_dim + action_shape[0],
            1,
            hidden_units,
        )
        

        # feat(t) = Encoder(x(t))
        self.encoder = Encoder(state_shape[0], ometer_shape[0]*ometer_shape[1], feature_dim)
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            z1_dim + z2_dim,
            tgt_state_shape[0],
            std=np.sqrt(image_noise),
        )
        self.apply(initialize_weight)

    def sample_prior(self, actions_, z2_post_):
        # p(z1(0)) = N(0, I)
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # p(z1(t) | z2(t-1), a(t-1))
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        # Concatenate initial and consecutive latent variables
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return (z1_mean_, z1_std_)

    def sample_posterior(self, features_, actions_):
        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, actions_.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return (z1_mean_, z1_std_, z1_, z2_)

    def calculate_loss(self, state_, ometer_, tgt_state_, action_, reward_, done_, cost_):
        # Calculate the sequence of features.
        feature_ = self.encoder(state_, ometer_)

        # Sample from latent variable model.
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # Prediction loss of images.
        z_ = torch.cat([z1_, z2_], dim=-1)
        state_mean_, state_std_ = self.decoder(z_)
        state_noise_ = (tgt_state_ - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_tgt_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_image_tgt_ = -log_likelihood_tgt_.mean(dim=0).sum()
        loss_image = loss_image_tgt_
        

        # Prediction loss of rewards and costs.
        x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean_, reward_std_ = self.reward(x.view(B * S, X))
        reward_mean_ = reward_mean_.view(B, S, 1)
        reward_std_ = reward_std_.view(B, S, 1)
        reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
        log_likelihood_reward_ = (-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()
        
        cost_mean_, cost_std_ = self.cost(x.view(B * S, X))
        cost_mean_ = cost_mean_.view(B, S, 1)
        cost_std_ = cost_std_.view(B, S, 1)
        cost_noise_ = (cost_ - cost_mean_) / (cost_std_ + 1e-8)
        log_likelihood_cost_ = (-0.5 * cost_noise_.pow(2) - cost_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_cost = -log_likelihood_cost_.mul_(1 - done_).mean(dim=0).sum()

        #p = self.cost(x.view(B * S, X)).view(B, S, 1)
        #q = 1-p
        #weight_p = 100
        #binary_cost_ = torch.sign(cost_)
        #loss_cost = -30*(weight_p*binary_cost_*torch.log(p+1e-6) + (1-binary_cost_)*torch.log(q+1e-6)).mean(dim=0).sum()

        return loss_kld, loss_image, loss_reward, loss_cost

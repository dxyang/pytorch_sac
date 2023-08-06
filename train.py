#!/usr/bin/env python3
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import math
import os
import sys
from tap import Tap
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import hydra

from agent.sac import SACAgent
from pointmass_utils import DmcPointTwoWall, PointEnvTwoWall
from helpers import Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_theme(style="white") # for the plots
cm_normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
cm_mako = sns.color_palette("mako", as_cmap=True)

def generate_expert_data(work_dir: str = None, num_episodes: int = 100):
    env = PointEnvTwoWall()

    trajs = []
    for _ in range(num_episodes):
        obs = env.reset()
        states = [obs]
        for t in range(env._max_episode_steps):
            obs, reward, done, info = env.step(env.get_expert_action())
            states.append(obs)

        assert info["dist_to_goal"] < 5e-2

        all_states = np.array(states)
        trajs.append(all_states)

    file_name = 'pointmass_twowall_expert_data.pkl'
    if work_dir is not None:
        file_name = f"{work_dir}/pointmass_twowall_expert_data.pkl"
    pickle.dump(trajs, open(file_name, 'wb'))

def setup_maze_plot():
    wall_line_width = 5
    wall_line_color = 'goldenrod'

    plt.clf()
    plt.cla()
    plt.xlim(0, 6)
    plt.ylim(0, 3)
    plt.xticks([])
    plt.yticks([])
    wall_one = [[2, 3], [2, 1]]
    wall_two = [[4, 0], [4, 2]]
    plt.plot(
        [wall_one[0][0], wall_one[1][0]],
        [wall_one[0][1], wall_one[1][1]],
        linewidth=wall_line_width, color=wall_line_color,
    )
    plt.plot(
        [wall_two[0][0], wall_two[1][0]],
        [wall_two[0][1], wall_two[1][1]],
        linewidth=wall_line_width, color=wall_line_color,
    )
    ax = plt.gca()
    ax.spines['left'].set_linewidth(wall_line_width)
    ax.spines['left'].set_color(wall_line_color)
    ax.spines['right'].set_linewidth(wall_line_width)
    ax.spines['right'].set_color(wall_line_color)
    ax.spines['top'].set_linewidth(wall_line_width)
    ax.spines['top'].set_color(wall_line_color)
    ax.spines['bottom'].set_linewidth(wall_line_width)
    ax.spines['bottom'].set_color(wall_line_color)
    plt.tight_layout()


def eval_network_over_statespace(classifier, ranker, workdir, step, plot_ranking = False, replay_buffer = None):
    eps = 1e-8
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    if replay_buffer is None:
        xs = np.random.uniform(0, 6,  size=(40000, 1))
        ys = np.random.uniform(0, 3,  size=(40000, 1))
        points = np.concatenate([xs, ys], axis=1)
    else:
        points = torch.from_numpy(replay_buffer.obses[:len(replay_buffer)])
    states_tensor = torch.Tensor(points).float().to(device)
    class_vals, rank_vals = 1.0, 1.0
    with torch.no_grad():
        if classifier is not None:
            class_vals = torch.sigmoid(classifier(states_tensor)).cpu().detach().numpy()
            clamped_class_vals = np.clip(class_vals, eps, 1 - eps)
        if ranker is not None:
            rank_vals = torch.sigmoid(ranker(states_tensor)).cpu().detach().numpy()
            clamped_rank_vals = np.clip(rank_vals, eps, 1 - eps)
    # combined_vals = np.log(clamped_rank_vals) + np.log(clamped_class_vals) - np.log(1 - clamped_class_vals)
    combined_vals = class_vals * rank_vals

    step_str = str(step).zfill(6)

    if classifier is not None:
        setup_maze_plot()
        cs = plt.scatter(points[:, 0], points[:, 1], c=class_vals, cmap=cm_mako, norm=cm_normalize)
        plt.colorbar(cs)
        plt.savefig(f"{workdir}/classifier_{step_str}.png");

    if ranker is not None and plot_ranking:
        setup_maze_plot()
        cs = plt.scatter(points[:, 0], points[:, 1], c=rank_vals, cmap=cm_mako, norm=cm_normalize)
        plt.colorbar(cs)
        plt.savefig(f"{workdir}/ranking_{step_str}.png");

    if classifier is not None and ranker is not None:
        setup_maze_plot()
        cs = plt.scatter(points[:, 0], points[:, 1], c=combined_vals, cmap=cm_mako, norm=cm_normalize)
        plt.colorbar(cs)
        plt.savefig(f"{workdir}/combined_{step_str}.png")


class Workspace(object):
    def __init__(self, cfg, args):
        self.work_dir = os.getcwd()
        self.exp_dir = f"/home/dxyang/code/rewardlearning-vid/pytorch_sac/exps/{args.exp_name}"
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        print(f'workspace: {self.work_dir}')
        print(f'expdir: {self.exp_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = PointEnvTwoWall() #DmcPointTwoWall(gym_env)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = SACAgent(
            obs_dim=cfg.agent.params.obs_dim,
            action_dim=cfg.agent.params.action_dim,
            action_range=cfg.agent.params.action_range,
            device=cfg.agent.params.device,
            discount=cfg.agent.params.discount,
            init_temperature=cfg.agent.params.init_temperature,
            alpha_lr=cfg.agent.params.alpha_lr,
            alpha_betas=cfg.agent.params.alpha_betas,
            actor_lr=cfg.agent.params.actor_lr,
            actor_betas=cfg.agent.params.actor_betas,
            actor_update_frequency=cfg.agent.params.actor_update_frequency,
            critic_lr=cfg.agent.params.critic_lr,
            critic_betas=cfg.agent.params.critic_betas,
            critic_tau=cfg.agent.params.critic_tau,
            critic_target_update_frequency=cfg.agent.params.critic_target_update_frequency,
            batch_size=cfg.agent.params.batch_size,
            learnable_temperature=cfg.agent.params.learnable_temperature,
        )
        #hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

        # ============= Expert data  ================ #
        if args.generate_expert_trajs:
            generate_expert_data(self.exp_dir, args.num_expert_trajs)
        self.expert_trajs = pickle.load(open(f'{self.exp_dir}/pointmass_twowall_expert_data.pkl', 'rb'))

        if args.plot_expert_data:
            setup_maze_plot()
            for traj in self.expert_trajs:
                plt.plot(traj[:, 0], traj[:, 1])
            plt.savefig(f"{self.exp_dir}/expert_trajs.png")

        # ============= Define reward and classifier functions  ================ #
        lr = 1e-4
        hidden_depth = 3
        hidden_dim = 1000
        episode_length = self.env._max_episode_steps

        self.ranking_network = Policy(obs_dim=self.env.observation_space.shape[0], action_dim = 1, hidden_dim=hidden_dim, hidden_depth=hidden_depth)
        self.classifier_network = Policy(obs_dim=self.env.observation_space.shape[0], action_dim = 1, hidden_dim=hidden_dim, hidden_depth=hidden_depth)
        self.ranking_network.to(device)
        self.classifier_network.to(device)
        self.ranking_optimizer = optim.Adam(list(self.ranking_network.parameters()), lr=lr)
        self.classifier_optimizer = optim.Adam(list(self.classifier_network.parameters()), lr=lr)
        self.classifier_losses = []
        self.update_classifier_freq = 900
        self.first_time_done = False

        # ============= Train ranking function ================ #
        if args.train_ranking_function:
            num_iterations = 10_000
            batch_size = 256
            bce_with_logits_criterion = torch.nn.BCEWithLogitsLoss()
            loss_steps = []
            losses = []

            # Train the model with regular SGD
            running_loss = 0.0
            plot_freq = 100
            for step in range(num_iterations):
                self.ranking_optimizer.zero_grad()

                # Same trajectory labels
                t1_idx = np.random.randint(len(self.expert_trajs), size=(batch_size,)) # Indices of first trajectory
                t1_idx_pertraj = np.random.randint(episode_length, size=(batch_size,))
                t1_idx_pertraj_second = np.random.randint(episode_length, size=(batch_size,))
                labels = np.zeros((batch_size,))
                first_before = np.where(t1_idx_pertraj < t1_idx_pertraj_second)[0]
                labels[first_before] = 1.0
                t1_states_first_np = np.concatenate([self.expert_trajs[c_idx][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
                t1_states_second_np = np.concatenate([self.expert_trajs[c_idx][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj_second)])
                t1_states_first = torch.Tensor(t1_states_first_np).float().to(device)
                t1_states_second = torch.Tensor(t1_states_second_np).float().to(device)
                t1_labels = F.one_hot(torch.Tensor(labels).long().to(device), 2).float()

                logit1 = self.ranking_network(t1_states_first)
                logit2 = self.ranking_network(t1_states_second)
                logits = torch.cat([logit1,  logit2], dim=-1)
                ranking_loss = bce_with_logits_criterion(logits, t1_labels)

                ranking_loss.backward()
                self.ranking_optimizer.step()

                # print statistics
                running_loss += ranking_loss.item()
                if step % plot_freq == 0 and step > 0:
                    print(f'[{step}] loss: {running_loss / plot_freq}')
                    losses.append(running_loss/plot_freq)
                    loss_steps.append(step)
                    running_loss = 0.0

                    plt.clf(); plt.cla()
                    plt.plot(loss_steps, losses)
                    plt.tight_layout()
                    plt.savefig(f"{self.exp_dir}/ranking_training.png")

            print(f"saved ranking network!")
            torch.save(self.ranking_network.state_dict(), f"{self.exp_dir}/ranking_network.pt")

        print(f"loading ranking network!")
        self.ranking_network.load_state_dict(torch.load(f"{self.exp_dir}/ranking_network.pt"))
        self.ranking_network.eval()

        if args.replot_ranking_network:
            # plot ranking function over state space
            xs = np.random.uniform(0, 6,  size=(40000, 1))
            ys = np.random.uniform(0, 3,  size=(40000, 1))
            points = np.concatenate([xs, ys], axis=1)
            states_tensor = torch.Tensor(points).float().to(device)
            points = np.concatenate([xs, ys], axis=1)
            states_tensor = torch.Tensor(points).float().to(device)
            with torch.no_grad():
                rank_vals = torch.sigmoid(self.ranking_network(states_tensor)).cpu().detach().numpy()

            setup_maze_plot()
            cs = plt.scatter(points[:, 0], points[:, 1], c=rank_vals,  cmap=cm_mako, norm=cm_normalize)
            plt.colorbar(cs)
            if not os.path.exists(f"{self.exp_dir}/full_ss"):
                os.makedirs(f"{self.exp_dir}/full_ss")
            plt.savefig(f"{self.exp_dir}/full_ss/ranking.png")

        if args.just_gail:
            self.ranking_network = None
        if args.just_ranking:
            self.classifier_network = None

    def evaluate(self, step):
        average_episode_rewards = []
        all_trajs = []
        for episode in range(self.cfg.num_eval_episodes):
            traj = []
            obs = self.env.reset()
            traj.append(obs)
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                traj.append(obs)
                self.video_recorder.record(self.env)
                episode_reward += reward

            all_trajs.append(copy.deepcopy(np.array(traj)))
            average_episode_rewards.append(episode_reward)
            self.video_recorder.save(f'{self.step}.mp4')
        reward_mean = np.mean(average_episode_rewards)
        reward_std = np.std(average_episode_rewards)
        self.logger.log('eval/episode_reward', reward_mean,
                        self.step)
        self.logger.dump(self.step)

        setup_maze_plot()
        for traj in all_trajs:
            plt.scatter(traj[:, 0], traj[:, 1])
        step_num = str(step).zfill(6)

        if not os.path.exists(f"{self.exp_dir}/eval"):
            os.makedirs(f"{self.exp_dir}/eval")
        plt.savefig(f"{self.exp_dir}/eval/{step_num}.png")

        eval_network_over_statespace(self.classifier_network, self.ranking_network, f"{self.exp_dir}/full_ss", step_num)
        eval_network_over_statespace(self.classifier_network, self.ranking_network, f"{self.exp_dir}/rb", step_num, plot_ranking=True, replay_buffer=self.replay_buffer)

        return reward_mean, reward_std

    def train_classifier_network(self, num_steps: int):
        self.classifier_network.train()

        batch_size = 256
        bce_with_logits_criterion = torch.nn.BCEWithLogitsLoss()
        for _ in range(num_steps):
            # sample from expert data
            t1_idx = np.random.randint(len(self.expert_trajs), size=(batch_size,)) # Indices of first trajectory
            t1_idx_pertraj = np.random.randint(self.env._max_episode_steps, size=(batch_size,))
            t1_states = np.concatenate([self.expert_trajs[c_idx][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_states_torch = torch.Tensor(t1_states).float().to(device)

            # sample from replay buffer
            obses, _, _, _, _, _ = self.replay_buffer.sample(batch_size)
            t2_states_torch = obses.float().to(device)

            labels_expert = torch.from_numpy(np.ones((batch_size,))).float().to(device)
            labels_negatives = torch.from_numpy(np.zeros((batch_size,))).float().to(device)

            states = torch.cat([t1_states_torch, t2_states_torch])
            labels = torch.cat([labels_expert, labels_negatives])

            self.classifier_optimizer.zero_grad()

            logits = self.classifier_network(states).squeeze()
            classify_loss = bce_with_logits_criterion(logits, labels)

            classify_loss.backward()
            self.classifier_optimizer.step()

            self.classifier_losses.append(classify_loss.item())

            plt.clf(); plt.cla()
            plt.plot(self.classifier_losses)
            plt.tight_layout()
            plt.savefig(f"{self.exp_dir}/classifier_losses.png")

        self.classifier_network.eval()

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        eval_reward_means = []
        eval_reward_stds = []
        eval_steps = []
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0 and self.step > self.cfg.num_seed_steps:
                    self.logger.log('eval/episode', episode, self.step)
                    reward_mean, reward_std = self.evaluate(self.step)
                    eval_reward_means.append(reward_mean)
                    eval_reward_stds.append(reward_std)
                    eval_steps.append(self.step)

                    # save and plot eval rewards
                    if self.step % (self.cfg.eval_frequency * 5) == 0:
                        eval_data = {
                            "reward_means": eval_reward_means,
                            "reward_stds": eval_reward_stds,
                            "steps": eval_steps,
                        }
                        pickle.dump(eval_data, open(f"{self.exp_dir}/eval_rewards.pkl", 'wb'))
                        plt.clf(); plt.cla()
                        plot_means = np.array(eval_reward_means)
                        plot_stds = np.array(eval_reward_stds)
                        plt.plot(eval_steps, plot_means)
                        plt.fill_between(eval_steps, plot_means - plot_stds, plot_means + plot_stds, alpha=0.2)
                        plt.tight_layout()
                        plt.savefig(f"{self.exp_dir}/eval_rewards.png")


                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                if not self.first_time_done:
                    self.env.set_classifier_network(self.classifier_network)
                    self.env.set_ranking_network(self.ranking_network)
                    if self.classifier_network is not None:
                        self.train_classifier_network(num_steps=100)
                    self.first_time_done = True

                if self.step % self.update_classifier_freq == 0:
                    if self.classifier_network is not None:
                        self.train_classifier_network(num_steps=20)

                self.agent.update(self.replay_buffer, self.logger, self.step, self.classifier_network, self.ranking_network)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

class ArgumentParser(Tap):
    exp_name: str

    just_gail: bool = False
    just_ranking: bool = False

    generate_expert_trajs: bool = False
    num_expert_trajs: int = 20
    plot_expert_data: bool = True

    train_ranking_function: bool = False
    replot_ranking_network: bool = True


def main():
    args = ArgumentParser().parse_args()

    from hydra import compose, initialize
    initialize(config_path="config")
    cfg = compose(config_name="train")

    workspace = Workspace(cfg, args)
    workspace.run()

if __name__ == '__main__':
    main()

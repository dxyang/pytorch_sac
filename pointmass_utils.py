
import copy
from dm_env import StepType, specs
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from helpers import doIntersect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# pointmass env - starts at center (0,0) and goes to goal randomly sampled from x, y (-1, 1)
class PointEnvStandard(gym.Env):
    def __init__(self, goal_center: np.ndarray = None, return_img_obs: bool = False, do_random_start: bool = False, do_random_goal: bool = False):
        self.pos = np.array([0., 0.])
        self.goal = np.array([-1., -1.])
        self.max_vel = 0.2
        if do_random_goal:
            self.observation_space = gym.spaces.Box(
                -np.inf * np.ones(4), np.inf * np.ones(4)
            )
        else:
            self.observation_space = gym.spaces.Box(
                -np.inf * np.ones(2), np.inf * np.ones(2)
            )

        self.goal_center = goal_center

        self.action_space = gym.spaces.Box(-np.ones(2), np.ones(2))
        self.action_space.seed(0)
        self.observation_space.seed(0)

        self.curr_step = 0
        self.episode_length = 100

        self.do_random_start = do_random_start
        self.do_random_goal = do_random_goal

        self.override_reward = False
        self.ranking_network = None
        self.classifier_network = None
        self.eps = 1e-5

        # image specific
        self.return_img_obs = return_img_obs
        self.img_size = 84
        self.ball_size= 5
        self.bucket_size = (10.5 - (-10.5)) / float(self.img_size)

    def reset(self):
        self.curr_step = 0
        self.pos = np.zeros((2,))
        self.goal = np.zeros((2,))

        if self.goal_center is not None:
            self.goal = self.goal_center

        if self.do_random_start:
            self.pos = np.random.uniform(-9, 9, size=(2,))

        if self.do_random_goal:
            self.goal = np.random.uniform(-9, 9, size=(2,))

        self.delta_vector = self.goal - self.pos

        return self.get_obs()

    def get_expert_action(self, add_noise: bool = False):
        delta_vector = self.goal - self.pos
        steps_remaining = self.episode_length - self.curr_step
        assert steps_remaining != 0
        expert_action = 1.0 / steps_remaining * delta_vector

        if add_noise:
            expert_action_magnitude = np.linalg.norm(expert_action)

            # add at most 50% of the action magnitude noise
            expert_action += np.random.normal(size=(2)) * expert_action_magnitude * 1.0

        return expert_action / self.max_vel

    def get_obs(self):
        if self.return_img_obs:
            img = self.render().copy()
            img_goal = self.render_goal().copy()
            return np.array([img, img_goal])
        else:
            if self.do_random_goal:
                return copy.deepcopy(np.concatenate([self.pos, self.goal]))
            else:
                return copy.deepcopy(self.pos)

    def render(self):
        img = 0*np.ones((self.img_size, self.img_size, 3), dtype=np.uint8)
        curr_pos = (self.pos - (-10.5)) // self.bucket_size
        img[int(curr_pos[0]):int(curr_pos[0]) + self.ball_size,
            int(curr_pos[1]):int(curr_pos[1]) + self.ball_size] = np.array([1., 0., 0.])
        img = np.transpose(img, [2, 0, 1])
        return img

    def render_goal(self):
        img = 0*np.ones((self.img_size, self.img_size, 3), dtype=np.uint8)
        curr_goal = (self.goal - (-10.5)) // self.bucket_size
        img[int(curr_goal[0]):int(curr_goal[0]) + self.ball_size,
            int(curr_goal[1]):int(curr_goal[1]) + self.ball_size] = np.array([1., 0., 0.])
        img = np.transpose(img, [2, 0, 1])
        return img

    def step(self, action: np.ndarray):
        self.pos += self.max_vel*action
        self.pos = np.clip(self.pos, -10., 10.)
        reward = -np.linalg.norm((self.pos - self.goal))
        og_reward = reward

        obs = self.get_obs()

        if self.override_reward:
            curr_obs = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                ranking_val, classify_val = 1.0, 1.0
                if self.ranking_network is not None:
                    ranking_val= torch.sigmoid(self.ranking_network(curr_obs)).cpu().item()
                if self.classifier_network is not None:
                    classify_val = torch.sigmoid(self.classifier_network(curr_obs)).cpu().item()

                ranking_val_clamped = np.clip(ranking_val, self.eps, 1 - self.eps) # numerical stability
                classify_val_clamped = np.clip(classify_val, self.eps, 1 - self.eps) # numerical stability

                if self.ranking_network is not None and self.classifier_network is not None:
                    # reward = np.log(ranking_val_clamped) + np.log(classify_val_clamped) - np.log(1 - classify_val_clamped)
                    reward = ranking_val * classify_val
                elif self.ranking_network is not None:
                    reward = np.log(ranking_val_clamped)
                elif self.classifier_network is not None:
                    reward = np.log(classify_val_clamped) - np.log(1 - classify_val_clamped)
                else:
                    # you want to override reward but don't say how
                    assert False

        self.curr_step += 1
        done = self.curr_step == self.episode_length

        return obs, reward, done, {"og_reward": og_reward}

    # def reward_fn(self, state):
    #     return -torch.linalg.norm((state - torch.Tensor(self.goal).to(device)), dim=-1)

    def set_ranking_network(self, ranking_network):
        self.ranking_network = ranking_network

    def set_classifier_network(self, classifier_network):
        self.classifier_network = classifier_network

class WaypointController():
    def __init__(self, waypoints, stepsize, threshold=5e-2, add_noise=True):
        self.waypoints = waypoints
        self.curr_waypoint = 0
        self.stepsize = stepsize
        self.threshold = threshold
        self.add_noise = add_noise

    def action(self, state):
        if self.curr_waypoint == len(self.waypoints):
            return np.array([0., 0.])

        vector = self.waypoints[self.curr_waypoint] - state
        distance = np.linalg.norm(vector)
        if distance <= self.threshold:
            # print(f"reached waypoint {self.curr_waypoint}")
            self.curr_waypoint += 1
            if self.curr_waypoint == len(self.waypoints):
                return np.array([0., 0.])

        vector = self.waypoints[self.curr_waypoint] - state
        distance = np.linalg.norm(vector)

        norm_vec = vector / distance
        random_stepsize = np.random.uniform() * self.stepsize

        noise = 0.1 * np.random.randn(2)

        return random_stepsize * norm_vec + (self.add_noise * noise)


class PointEnvTwoWall(gym.Env):
    def __init__(self, noisy_start = True):
        self.waypoints = [
            [1, 0.5],
            [3, 0.5],
            [3, 2.5],
            [5, 2.5],
            [5, 0.5]
        ]
        self.max_vel = 0.3
        self._max_episode_steps = 300
        self.noisy_start = noisy_start

        # the bounds are slightly modified extended for the sake of preventing
        # the agent frome exploiting floating point cheating
        self.wall_one = [[2, 3.5], [2, 1]]
        self.wall_two = [[4, -0.5], [4, 2]]

        self.reset()

        self.observation_space = gym.spaces.Box(-np.inf * np.ones(2), np.inf * np.ones(2))
        self.action_space = gym.spaces.Box(-np.ones(2), np.ones(2))
        self.action_space.seed(0)
        self.observation_space.seed(0)

        self.classifier = None
        self.ranker = None
        self.eps = 1e-5

    def set_classifier_network(self, classifier: torch.nn.Module):
        self.classifier = classifier

    def set_ranking_network(self, ranking: torch.nn.Module):
        self.ranker = ranking

    def step(self, action: np.ndarray):
        new_pos = self.pos + self.max_vel*action

        # do wall clamping
        intersects_wall_one = doIntersect(self.pos, new_pos, self.wall_one[0], self.wall_one[1])
        intersects_wall_two = doIntersect(self.pos, new_pos, self.wall_two[0], self.wall_two[1])
        if intersects_wall_one or intersects_wall_two:
            self.pos = self.pos
        else:
            self.pos = new_pos

        self.pos[0] = np.clip(self.pos[0], 0.0, 6.0)
        self.pos[1] = np.clip(self.pos[1], 0.0, 3.0)

        obs = self.get_obs()

        self.curr_step += 1
        dist_to_goal = np.linalg.norm(self.pos - self.goal)

        if self.classifier is None and self.ranker is None:
            if dist_to_goal < 5e-2:
                reward = 1.0
            else:
                reward = 0.0
        else:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            with torch.no_grad():
                ranking_val, classify_val = 1.0, 1.0
                if self.ranker is not None:
                    ranking_val= torch.sigmoid(self.ranker(obs_tensor)).cpu().item()
                if self.classifier is not None:
                    classify_val = torch.sigmoid(self.classifier(obs_tensor)).cpu().item()

                ranking_val_clamped = np.clip(ranking_val, self.eps, 1 - self.eps) # numerical stability
                classify_val_clamped = np.clip(classify_val, self.eps, 1 - self.eps) # numerical stability

                if self.ranker is not None and self.classifier is not None:
                    # reward = np.log(ranking_val_clamped) + np.log(classify_val_clamped) - np.log(1 - classify_val_clamped)
                    reward = ranking_val * classify_val
                elif self.ranker is not None:
                    # reward = np.log(ranking_val_clamped)
                    reward = ranking_val
                elif self.classifier is not None:
                    # reward = np.log(classify_val_clamped) - np.log(1 - classify_val_clamped)
                    reward = classify_val
                else:
                    # you want to override reward but don't say how
                    assert False

        # reward = -np.linalg.norm((self.pos - self.goal))
        og_reward = 0.0

        done = self.curr_step == self._max_episode_steps #or dist_to_goal < 5e-2

        return obs, reward, done, {"og_reward": og_reward, "dist_to_goal": dist_to_goal}

    def get_expert_action(self):
        return self.ctlr.action(self.pos)

    def get_obs(self):
        return copy.deepcopy(self.pos)

    def reset(self):
        self.curr_step = 0
        self.pos = np.array([1., 2.5])
        random_noise = (np.random.rand(2) - 0.5) / 2
        if self.noisy_start:
            self.pos += random_noise
        self.goal = np.array([5., 0.5])
        self.ctlr = WaypointController(self.waypoints, self.max_vel)
        return self.get_obs()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = PointEnvTwoWall()
    num_episodes = 200

    trajs = []
    lengths = []
    for _ in range(num_episodes):
        obs = env.reset()
        states = [obs]
        for t in range(10_000):
            obs, reward, done, info = env.step(env.get_expert_action())
            if info["dist_to_goal"] < 0.1:
                lengths.append(t)
                break
            states.append(obs)

        all_states = np.array(states)
        trajs.append(all_states)

    print(np.mean(lengths), np.std(lengths))
    plt.figure()

    # configure box
    plt.xlim(0, 6)
    plt.ylim(0, 3)
    wall_one = [[2, 3], [2, 1]]
    wall_two = [[4, 0], [4, 2]]
    plt.plot(
        [wall_one[0][0], wall_one[1][0]],
        [wall_one[0][1], wall_one[1][1]]
    )
    plt.plot(
        [wall_two[0][0], wall_two[1][0]],
        [wall_two[0][1], wall_two[1][1]]
    )
    # plt.scatter(all_states[:, 0], all_states[:, 1])
    for traj in trajs:
        plt.plot(traj[:, 0], traj[:, 1])
    plt.show()

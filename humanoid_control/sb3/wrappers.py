import time
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

class MocapTrackingVecMonitor(VecMonitor):
    """
    A VecMonitor that additionally monitors the normalized episode return and length.
    """
    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])

        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                t_start, t_end = info['start_time_in_clip'], info['time_in_clip']
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_length_norm = (t_end - t_start) / (info['last_time_in_clip'] - t_start)
                episode_return_norm = episode_return / episode_length * episode_length_norm

                episode_info = dict(
                    r=episode_return,
                    r_norm=episode_return_norm,
                    l=episode_length,
                    l_norm=episode_length_norm,
                    t=round(time.time() - self.t_start, 6)
                )
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info['episode'] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos

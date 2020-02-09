import numpy as np
import torch


class VecEnv(object):

    def __init__(self, make_env_fn, num_env):
        self.envs = tuple(make_env_fn() for _ in range(num_env))

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, item):
        return self.envs[item]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.envs):
            result = self.envs[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def reset(self):
        obs = [env.reset() for env in self.envs]
        obs = _flatten_list_of_dicts(obs)
        return obs

    def step(self,
             actions,
             core_hidden):

        assert len(self.envs) == len(actions) == len(core_hidden)
        results = []
        for env, a, h in zip(self.envs, actions, core_hidden):
            results.append(env.step(a, h))
        results = _flatten_list_of_dicts(results)
        return results

    # Call this at the end of training:
    def close(self):
        for env in self.envs:
            env.close()


def _flatten_list_of_dicts(list_of_dicts):
    assert isinstance(list_of_dicts, (list, tuple))
    assert len(list_of_dicts) > 0
    if isinstance(list_of_dicts[0], dict):
        keys = list_of_dicts[0].keys()
        result = {}
        for k in keys:
            if isinstance(list_of_dicts[0][k], np.ndarray):
                result[k] = np.stack([o[k] for o in list_of_dicts])
            elif isinstance(list_of_dicts[0][k], torch.Tensor):
                result[k] = torch.stack([o[k] for o in list_of_dicts])
            else:
                result[k] = [o[k] for o in list_of_dicts]
        return result
    else:
        return np.stack(list_of_dicts)


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])
    return [l__ for l_ in l for l__ in l_]
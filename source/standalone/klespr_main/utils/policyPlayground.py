import torch


@torch.no_grad()
def play_policy_shared(policy, skrl_env):

    policy.to(skrl_env.device)

    states, _ = skrl_env.reset()

    for i in range(100):

        actions = policy.act({"states": states}, role="policy")[0]

        next_states, rewards, terminated, truncated, infos = skrl_env.step(actions)

        states = next_states

import torch


@torch.no_grad()
def play_policy_shared(policy, skrl_env):

    policy.to(skrl_env.device)

    # we want to check how KL Divergance calculations work with our policy

    ppo_policy = policy

    states, _ = skrl_env.reset()

    all_kls = []

    logRatioSum = 0

    for i in range(100):

        actions, log_prob, _ = policy.act({"states": states}, role="policy")

        _, ppo_log_prob, _ = ppo_policy.act({"states": states, "taken_actions": actions}, role="policy")

        next_states, rewards, terminated, truncated, infos = skrl_env.step(actions)

        states = next_states

        log_ratio = log_prob - ppo_log_prob

        kl_divergence = ((torch.exp(log_ratio) - 1) - log_ratio).mean()

        all_kls.append(kl_divergence)

        logRatioSum += log_ratio.sum().item()

    kl = torch.tensor(all_kls).mean()

    print(kl)


@torch.no_grad()
def play_policy_mlp(policy, skrl_env):

    policy.to(skrl_env.device)

    # we want to check how KL Divergance calculations work with our policy

    states, _ = skrl_env.reset()

    for i in range(100):

        actions = policy.act({"states": states}, role="policy")[0]

        print(actions)

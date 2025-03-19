from adacth import adact_h


def RegORL(dataset, epsilon, delta, actions, observations, rewards, horizon, upper=0):
    if upper == 0:
        upper = (len(actions)*len(observations))**horizon * 2
    random.shuffle(dataset)
    split = len(dataset)//2
    d1 = dataset[:split]
    d2 = dataset[split:]
    states, transitions = adact_h(d1, delta, horizon, actions, observations, rewards, chi_index=1)

    with open('states.txt', 'w') as f:
        for statet in states:
            for state in statet:
                f.write(f'"{state}"\n')

    with open('transitions.txt', 'w') as f:
        for key, value in transitions.items():
            f.write(f'[\"{key[0]}\", \"{key[1]}\", \"{value}\"]\n')

    print("States:")
    for statet in states:
        for state in statet:
            print('"'+str(state)+'"')
    print("Transitions:")
    for key, value in transitions.items():
        print('"'+str(key[0])+'"', '"'+str(value)+'"', '"'+str(key[1])+'"')
    print("")
    d2_new = transform(d2, transitions)
    states = set().union(*states)
    policy = offlineRL(d2_new, epsilon, delta/2, states, actions, rewards, horizon)
    with open('policy.txt', 'w') as f:
        for h in policy:
            for s in policy[h]:
                f.write(f'[\"{h}\", \"{s}\", \"{policy[h][s]}\"]\n')
    return policy

def transform(dataset, transitions):
    d = []
    for episode in dataset:
        new_episode = []
        state = 'u0'
        h = 0
        for step in episode:
            new_state = transitions[(state, (step[0], step[1]))]
            new_step = (state, step[0], step[2], new_state)
            new_episode.append(new_step)
            state = new_state
            h += 1
        d.append(new_episode)
    return d


def offlineRL(d2, epsilon, delta, states, actions, rewards, horizon):
    return VILCB(d2, epsilon, delta, states, actions, rewards, horizon)

import random
import math
import numpy as np
from collections import defaultdict

# def subVILCB(data, epsilon, delta):
#     random.shuffle(data)
#     split = len(data)//2
#     d1 = data[:split]
#     d2 = data[split:]
#     nmain(s)
#     naux(s)

#     for s in states:
#         for h in range(horizon):
#             ntrim(s) = max(naux(s)- 10*math.sqrt(naux(s)*math.log(horizon*len(states)/delta)), 0)
    
#     d(0)= d(trim)
#     return VILCB(d(0), epsilon, delta)

def compute_stateactions(dataset):
    stateactions = defaultdict(set)
    for episode in dataset:
        for step in episode:
            stateactions[step[0]].add(step[1])
    return stateactions

def compute_empirical_kernel(dataset, h, stateactions, rewards):
    """
    For a given time step h, compute the empirical transition kernel.
    Returns:
      - P_hat: a dict mapping (s, a) to a dictionary over next states s' with probabilities.
      - counts: a dict mapping (s, a) to count of transitions at time h.
    If a given (s, a) pair is not observed, use a uniform distribution over states.
    """
    counts = defaultdict(int)
    next_state_counts = defaultdict(lambda: defaultdict(int))
    r_func = defaultdict(lambda: [0,0])
    maxr = max(rewards)
    minr = min(rewards)
    # Filter dataset for transitions at step h
    for episode in dataset:
        (s, a, r, s_next) = episode[h]
        counts[(s, a)] += 1
        next_state_counts[(s, a)][s_next] += 1
        r_func[(s, a, h)][0] += (r - minr)/(maxr - minr)
        r_func[(s, a, h)][1] += 1

    P_hat = {}
    for s in stateactions:
        for a in stateactions[s]:
            key = (s, a)
            count = counts.get(key, 0)
            if count > 0:
                # Build probability distribution for next state
                P_hat[key] = {s_next: cnt / count for s_next, cnt in next_state_counts[key].items()}
            else:
                # If no sample available, use uniform probability over states
                P_hat[key] = {s_next: 1/len(stateactions) for s_next in stateactions}
    return P_hat, counts, r_func

def variance_of_value(P_dist, V_next):
    """
    Compute variance of V_next under distribution P_dist.
    V_next is a dict mapping states to their value.
    """
    exp_val = sum(P_dist[s] * V_next[s] for s in P_dist)
    exp_val_sq = sum(P_dist[s] * (V_next[s] ** 2) for s in P_dist)
    return max(exp_val_sq - exp_val**2, 0.0)

def VILCB(dataset, epsilon, delta, states, actions, rewards, H, c_b=0.5):
    V = {H+1: {s: 0.0 for s in states}}
    Q = {}
    policy = {}
    N = len(dataset)
    
    stateactions = compute_stateactions(dataset)

    # Backward induction: for h = H, H-1, ..., 1
    for h in range(H, 0, -1):
        print('Loop', h)
        # Compute the empirical transition kernel P_hat for step h
        P_hat, counts, r = compute_empirical_kernel(dataset, h, stateactions, rewards)
        for rvalue in r:
            print(rvalue, r[rvalue])
        Q[h] = {}
        V[h] = {}
        policy[h] = {}
        
        # For each state-action pair, compute Q-value
        for s in stateactions:
            for a in stateactions[s]:
                key = (s, a)
                if s == "((((('u0', '1', '110'), '1', '101'), '1', '101'), '1', '101'), '1', '010')":
                    print(key,)
                N_sa = counts.get(key, 0)
                # Estimate expected value: sum_{s'} P_hat(s'|s,a) * V_{h+1}(s')
                exp_val = sum(P_hat[key][s_next] * V[h+1][s_next] for s_next in P_hat[key])
                
                # Compute variance under empirical distribution P_hat(s'|s,a)
                var_val = variance_of_value(P_hat[key], V[h+1])
                
                # Use total count at step h in the log term (could be refined)
                log_term = math.log(N* H / delta)
                
                if s == ((((('u0', '1', '110'), '1', '101'), '1', '101'), '1', '101'), '1', '010'):
                    #print([(P_hat[key][s_next], V[h+1][s_next], s_next) for s_next in P_hat[key]])
                    print(key, N_sa, exp_val, r[(s, a, h)], 'test')

                if N_sa > 0:
                    penalty = math.sqrt(c_b * log_term * var_val / N_sa) + (c_b * H * log_term) / N_sa
                    # Clip penalty at H
                    penalty = min(penalty, H)
                else:
                    # No data for (s,a) at step h: set penalty high so that Q becomes 0.
                    penalty = H
                
                # Q-value is the pessimistic estimate
                if r[(s, a, h)][1] == 0:
                    Q_val = 0.0
                else:
                    print(s,a,h,'sah')
                    
                    Q_val = r[(s, a, h)][0]/r[(s, a, h)][1] + exp_val - penalty
                    print(round(r[(s, a, h)][0]/r[(s, a, h)][1],2), exp_val, penalty, Q_val, "Q_val")
                    Q_val = max(Q_val, 0.0)  # enforce non-negativity
                Q[h][(s, a)] = Q_val
                
        # For each state, compute V and choose greedy action
        for s in states:
            best_a = None
            best_q = -float('inf')
            for a in stateactions[s]:
                q_val = Q[h][(s, a)]
                if q_val > best_q:
                    best_q = q_val
                    best_a = a
            best_q = max(best_q, 0)
            V[h][s] = best_q
            policy[h][s] = best_a
            
    return policy

if __name__ == "__main__":

    horizon = 5  # Maximum length of an episode
    delta = 0.01  # Failure probability

    dataset = []
    with open('tmaze25x5x1.txt', 'r') as f:
        for line in f:
            episode = eval(line.strip())
            for i in range(len(episode)-horizon):
                dataset.append(episode[i:i+horizon+1])

    actions = set()
    observations = set()
    rewards = set()
    for episode in dataset:
        for step in episode:
            actions.add(step[0])
            observations.add(step[1])
            rewards.add(step[2])
    print(actions, observations, rewards)

    output = RegORL(dataset, 0.1, delta, actions, observations, rewards, horizon)
    # for i in output:
    #     print(i)
    #     for j in output[i]:
    #         print(j, output[i][j])

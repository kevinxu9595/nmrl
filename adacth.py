import math
from collections import defaultdict

from multiprocessing import Pool, freeze_support
from functools import partial
import random
from timeit import default_timer as timer

def adact_h(dataset, delta, horizon, actions, observations, rewards, g_index = 3, chi_index=None):
    """
    ADACT-H algorithm for learning a minimal RDP from offline data.

    Args:
        dataset: List of episodes, where each episode is a sequence of (action, observation, reward) tuples.
        delta: Failure probability.
        horizon: Maximum length of an episode (H).
        actions: Set of possible actions (A).
        observations: Set of possible observations (O).

    Returns:
        states: Set of RDP states.
        transitions: Transition function mapping (state, action-observation) pairs to next states.
    """
    g = generate_g(g_index, actions, observations, rewards)
    if chi_index is None:
        all_chi = generate_chi(horizon, len(g))
    else:
        all_chi = generate_chi(min(horizon, chi_index), len(g))

    # Initialize variables
    u0 = "u0"  # Initial state
    U_states = []
    U_states.append({u0})
    tau_transitions = {}  # Transition function
    Z_suffixes = defaultdict(list)  # Suffixes associated with each state

    # Assign the dataset to the initial state
    Z_suffixes[u0] = dataset

    # Iterate over time steps
    for t in range(horizon+1):
        print("Loop",t)
        if chi_index is None:
            chi = all_chi[:horizon-t+1]
        else:
            chi = all_chi[:min(horizon-t, chi_index)+1]

        U_candidate_states = set()  # Candidate states at the next level
        next_suffixes = defaultdict(list)
        time1 = timer()
        # Generate candidates for the next states
        for s in U_states[t]:
            simplesuffixes = [x[1:] for x in Z_suffixes[s]]
            for (a, o) in [(a, o) for a in actions for o in observations]:
                candidate_state = (s,a,o)
                U_candidate_states.add(candidate_state)
                # Collect suffixes corresponding to the candidate state
                for episode in dataset:
                    if check_transition(s, episode, tau_transitions, t):
                        ao, suffix = episode[t][:2], episode[t + 1:]
                        if ao == (a, o) and (suffix in simplesuffixes or (suffix == [] and any(len(x)==x for x in Z_suffixes[s]))):
                            next_suffixes[candidate_state].append(suffix)
                if candidate_state not in next_suffixes:
                    U_candidate_states.remove(candidate_state)
        time2 = timer()
        print("Time taken to generate candidates:", time2-time1)
        uaom = max(next_suffixes, key= lambda x: len(next_suffixes[x]))
        promoted_states = {uaom}
        tau_transitions[(uaom[0], uaom[1:])] = uaom
        U_candidate_states.remove(uaom)
        print(uaom, "uaom,", U_candidate_states, "candidates")
        # Promote and merge candidate states
        for candidate in U_candidate_states:
            time1 = timer()
            is_promoted = True
            # Compare with already promoted states
            for existing_state in promoted_states:
                print(candidate, "compared to", existing_state)
                if not test_distinct(next_suffixes[candidate], next_suffixes[existing_state], delta, t, horizon, chi, g):
                    # Merge candidate into existing state
                    is_promoted = False
                    tau_transitions.update({(candidate[0], candidate[1:]): existing_state})
                    next_suffixes[existing_state].extend(next_suffixes[candidate])
                    break

            if is_promoted:
                promoted_states.add(candidate)
                tau_transitions.update({(candidate[0], candidate[1:]): candidate})
            time2 = timer()
            print("Time taken to compare:", time2-time1)

        # Update states and suffixes for the next iteration
        U_states.append(promoted_states)
        Z_suffixes = next_suffixes
        print(U_states, t)

    return U_states, tau_transitions

def generate_g(g_index, actions, observations, rewards):
    ga = set([tuple((a, o, r) for o in observations for r in rewards) for a in actions])
    go = set([tuple((a, o, r) for a in actions for r in rewards) for o in observations])
    gr = set([tuple((a, o, r) for a in actions for o in observations) for r in rewards])
    g1 = ga.union(go).union(gr)
    if g_index == 1:
        return g1
    gao = set([tuple((a, o, r) for r in rewards) for a in actions for o in observations])
    gar = set([tuple((a, o, r) for o in observations) for a in actions for r in rewards])
    gor = set([tuple((a, o, r) for a in actions) for o in observations for r in rewards])
    g2 = g1.union(gao).union(gar).union(gor)
    if g_index == 2:
        return g2
    gaor = set([tuple([(a, o, r)]) for a in actions for o in observations for r in rewards])
    g3 = g2.union(gaor)
    return g3

def check_transition(state, episode, transitions, t):
    if len(episode) < t:
        return False
    current = "u0"
    for i in episode[:t]:
        current = transitions[(current,i[:2])]
    return current == state

def generate_chi(l, g):
    chi = []
    for k in range(1,l+1):
        newg = []
        for i in range(g**k):
            somelists = tuple(i//g**(x-1) % g for x in range(k,0, -1))
            newg.append(somelists)
        chi.append(newg)
    return chi

def test_distinct(suffixes1, suffixes2, delta, t, horizon, chi, g):
    """
    Statistical test to determine if two states are distinct.

    Args:
        suffixes1: List of suffixes for the first state.
        suffixes2: List of suffixes for the second state.
        delta: Failure probability.
        t: Current time step.
        actions: Set of possible actions.
        observations: Set of possible observations.

    Returns:
        True if the states are distinct, False otherwise.
    """
    # Calculate Lp-infinity distance between the two sets of suffixes
    distance = l_x_distance(suffixes1, suffixes2, chi, g)
    # Threshold for distinguishing states
    threshold = math.sqrt(2 * math.log(2 * sum(len(x) for x in chi) / delta) / min(len(suffixes1), len(suffixes2)))
    print(distance, threshold, distance >= threshold)
    return distance >= threshold


def l_x_distance(suffixes1, suffixes2, chi, g):
    """
    Compute the L_x distance between two sets of suffixes.

    Args:
        suffixes1: List of suffixes for the first state.
        suffixes2: List of suffixes for the second state.

    Returns:
        The Lp-infinity distance between the two sets of suffixes.
    """
    # Convert suffixes to frequency distributions
    probs1 = defaultdict(int)
    probs2 = defaultdict(int)
    arrayg = list(g)
    for s in suffixes1:
        probs1[tuple(s)] += 1
    for s in suffixes2:
        probs2[tuple(s)] += 1
    for key in probs1:
        probs1[key] /= len(suffixes1)
    for key in probs2:
        probs2[key] /= len(suffixes2)
    max_diff = 0
    args = [probs1,probs2]
    func = partial(getprobs, arrayg, chi)
    results = [getprobs(arrayg, chi, x) for x in args]
    # for p, o, c in zip(results[0], results[1], chi[0]):
    #     print(p, o, arrayg[c[0]])
    max_diff = max([abs(p-o) for p, o in zip(results[0], results[1])])
    return max_diff

def getprobs(arrayg, chi, probs_i):
    probs = []
    for time in chi:
        for x in time:
            prob = 0
            for s in probs_i:
                i = 0
                for aor in s:
                    if aor in arrayg[x[i]]:
                        i +=1
                    if i == len(x):
                        prob += probs_i[tuple(s)]
                        break
            probs.append(prob)
    return probs



if __name__ == "__main__":
    horizon = 5  # Maximum length of an episode
    delta = 0.01  # Failure probability

    dataset = []
    with open('tmaze25x5x1.txt', 'r') as f:
        for line in f:
            episode = eval(line.strip())
            for i in range(len(episode)-horizon):
                dataset.append(episode[i:i+horizon+1])

    # actions = [0, 1]  # Two possible actions
    # observations = ['x', 'y']  # Three possible observations
    # rewards = [0, 1]
    # data = [
    #     [(0,"x",0), 
    #     (rewards[j//4 % 2], observations[i//2 % 2], 1-rewards[j//4 % 2]), 
    #     (rewards[j//2 % 2], observations[i % 2], 1-rewards[j//2 % 2]), 
    #     (rewards[j % 2], 'o0', 0)] for i in range(2*2) for j in range(2*2*2)
    # ] + [
    #     [(0,"y",0), 
    #     (rewards[j//4 % 2], observations[i//2 % 2], rewards[j//4 % 2]), 
    #     (rewards[j//2 % 2], observations[i % 2], rewards[j//2 % 2]), 
    #     (rewards[j % 2], 'o0', 0)] for i in range(2*2) for j in range(2*2*2)
    # ] 
    # dataset = []
    # for i in range(1000):
    #     dataset.append(random.choice(data))
    # with open('basic.txt', 'w') as f:
    #     for episode in dataset:
    #         f.write(str(episode) + '\n')

    actions = set()
    observations = set()
    rewards = set()
    for episode in dataset:
        for step in episode:
            actions.add(step[0])
            observations.add(step[1])
            rewards.add(step[2])
    print(actions, observations, rewards)

    

    states, transitions = adact_h(dataset, delta, horizon, actions, observations, rewards, chi_index=1)
    print("States:")
    for statet in states:
        for state in statet:
            print('"'+str(state)+'"')
    print("Transitions:")
    for key, value in transitions.items():
        print('"'+str(key[0])+'"', '"'+str(value)+'"', '"'+str(key[1])+'"')
import math
from collections import defaultdict

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
            for episode in dataset:
                if check_transition(s, episode, tau_transitions, t):
                    a, o, suffix = episode[t][0], episode[t][1], episode[t + 1:]
                    if (suffix in simplesuffixes or (suffix == [] and any(len(x)==x for x in Z_suffixes[s]))):
                        next_suffixes[(s,a,o)].append(suffix)
        U_candidate_states = set(next_suffixes.keys())

            # for (a, o) in [(a, o) for a in actions for o in observations]:
            #     candidate_state = (s,a,o)
            #     U_candidate_states.add(candidate_state)
            #     # Collect suffixes corresponding to the candidate state
            #     for episode in dataset:
            #         if check_transition(s, episode, tau_transitions, t):
            #             ao, suffix = episode[t][:2], episode[t + 1:]
            #             if ao == (a, o) and (suffix in simplesuffixes or (suffix == [] and any(len(x)==x for x in Z_suffixes[s]))):
            #                 next_suffixes[candidate_state].append(suffix)
            #     if candidate_state not in next_suffixes:
            #         U_candidate_states.remove(candidate_state)
            
        time2 = timer()
        print("Time taken to generate candidates:", time2-time1)
        uaom = max(next_suffixes, key= lambda x: len(next_suffixes[x]))
        promoted_states = {uaom}
        tau_transitions[(uaom[0], uaom[1:])] = uaom
        U_candidate_states.remove(uaom)
        print(uaom, "uaom,", len(U_candidate_states), "candidates")
        # Promote and merge candidate states
        time1 = timer()

        probs = suffix_probs(promoted_states, U_candidate_states, next_suffixes)

        for candidate in U_candidate_states:
            
            is_promoted = True
            # Compare with already promoted states
            for existing_state in promoted_states:
                if not test_distinct(probs[candidate], probs[existing_state], delta, t, horizon, chi, g):
                    # Merge candidate into existing state
                    is_promoted = False
                    tau_transitions.update({(candidate[0], candidate[1:]): existing_state})
                    next_suffixes[existing_state].extend(next_suffixes[candidate])
                    probs = update_suffix_probs(candidate, existing_state, probs)
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
    actions = set(actions)
    actions.discard('a0')
    observations.discard('o0')
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

def test_distinct(probs1, probs2, delta, t, horizon, chi, g):
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
    distance = l_x_distance(probs1[0], probs2[0], chi, g)
    # Threshold for distinguishing states
    threshold = math.sqrt(2 * math.log(2 * sum(len(x) for x in chi) / delta) / min(probs1[1], probs2[1]))
    return distance >= threshold

def suffix_probs(promoted_states, U_candidate_states, next_suffixes):
    probs = {}
    for state in promoted_states:
        probs[state] = [defaultdict(int), len(next_suffixes[state])]
        for suffix in next_suffixes[state]:
            probs[state][0][tuple(suffix)] += 1/len(next_suffixes[state])
    for state in U_candidate_states:
        probs[state] = [defaultdict(int), len(next_suffixes[state])]
        for suffix in next_suffixes[state]:
            probs[state][0][tuple(suffix)] += 1/len(next_suffixes[state])
    return probs

def update_suffix_probs(candidate, existing_state, probs):
    # Update the suffix probabilities for the existing state
    for suffix in probs[candidate][0]:
        probs[existing_state][0][tuple(suffix)] += (probs[candidate][0][tuple(suffix)]*probs[existing_state][1] - 
                                                    probs[existing_state][0][tuple(suffix)]*probs[candidate][1]) / (probs[existing_state][1]*(probs[existing_state][1] + probs[candidate][1]))
    # Update the count of suffixes for the existing state
    probs[existing_state][1] += probs[candidate][1]
    # Remove the candidate state from the dictionary    
    del probs[candidate]
    return probs

def l_x_distance(probs1, probs2, chi, g):
    """
    Compute the L_x distance between two sets of suffixes.

    Args:
        suffixes1: List of suffixes for the first state.
        suffixes2: List of suffixes for the second state.

    Returns:
        The Lp-infinity distance between the two sets of suffixes.
    """
    # Convert suffixes to frequency distributions
    arrayg = list(g)
    max_diff = 0
    args = [probs1,probs2]
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
    horizon = 15  # Maximum length of an episode
    delta = 0.01  # Failure probability

    dataset = []
    with open('data\\minihallx15.txt', 'r') as f:
        for line in f:
            episode = eval(line.strip())
            dataset.append(episode[:horizon+1])

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
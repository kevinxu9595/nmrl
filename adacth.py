import copy
import math, re
from collections import defaultdict

from multiprocessing import Pool, freeze_support
from functools import partial


def adact_h(dataset, delta, horizon, actions, observations, rewards, policy):
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
    ga = set([tuple((a, o, r) for o in observations for r in rewards) for a in actions])
    go = set([tuple((a, o, r) for a in actions for r in rewards) for o in observations])
    gr = set([tuple((a, o, r) for a in actions for o in observations) for r in rewards])
    g1 = ga.union(go).union(gr)
    gao = set([tuple((a, o, r) for r in rewards) for a in actions for o in observations])
    gar = set([tuple((a, o, r) for o in observations) for a in actions for r in rewards])
    gor = set([tuple((a, o, r) for a in actions) for o in observations for r in rewards])
    g2 = g1.union(gao).union(gar).union(gor)
    gaor = set([tuple([(a, o, r)]) for a in actions for o in observations for r in rewards])
    g3 = g2.union(gaor)

    # Initialize variables
    u0 = "u0"  # Initial state
    U_states = []
    U_states.append({u0})
    tau_transitions = {}  # Transition function
    Z_suffixes = defaultdict(list)  # Suffixes associated with each state

    # Assign the dataset to the initial state
    Z_suffixes[u0] = dataset

    # Iterate over time steps
    for t in range(horizon):
        print("Loop",t)
        chi = generate_chi(horizon-t, len(g3))

        U_candidate_states = set()  # Candidate states at the next level
        next_suffixes = defaultdict(list)

        # Generate candidates for the next states
        for s in U_states[t]:
            for (a, o) in [(a, o) for a in actions for o in observations]:
                candidate_state = (s,a,o)
                U_candidate_states.add(candidate_state)
                # Collect suffixes corresponding to the candidate state
                for episode in dataset:
                    if check_transition(s, episode, tau_transitions, t):
                        ao, suffix = episode[t][:2], episode[t + 1:]
                        if ao == (a, o) and (suffix in [x[1:] for x in Z_suffixes[s]] or (suffix == [] and any(len(x)==x for x in Z_suffixes[s]))):
                            next_suffixes[candidate_state].append(suffix)
                if candidate_state not in next_suffixes:
                    U_candidate_states.remove(candidate_state)
        # print(next_suffixes, "next")
        uaom = max(next_suffixes, key= lambda x: len(next_suffixes[x]))
        promoted_states = {uaom}
        tau_transitions[(uaom[0], uaom[1:])] = uaom
        U_candidate_states.remove(uaom)
        print(uaom, "uaom", U_candidate_states, "candidates")
        # Promote and merge candidate states
        for candidate in U_candidate_states:
            is_promoted = True
            # Compare with already promoted states
            for existing_state in promoted_states:
                print(candidate, ".", existing_state)
                if not test_distinct(next_suffixes[candidate], next_suffixes[existing_state], delta, t, actions, observations, horizon, chi, g3, policy):
                    # Merge candidate into existing state
                    is_promoted = False
                    print({(candidate[0], candidate[1:]): existing_state})
                    tau_transitions.update({(candidate[0], candidate[1:]): existing_state})
                    next_suffixes[existing_state].extend(next_suffixes[candidate])
                    break

            if is_promoted:
                promoted_states.add(candidate)
                tau_transitions.update({(candidate[0], candidate[1:]): candidate})
                print(candidate,"promoted")
            else:
                print(candidate,"merged")

        # Update states and suffixes for the next iteration
        U_states.append(promoted_states)
        Z_suffixes = next_suffixes
        print(U_states, t)

    return U_states, tau_transitions

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
        chi += newg
    print(len(chi))
    return chi

def test_distinct(suffixes1, suffixes2, delta, t, actions, observations, horizon, chi, g3, policy):
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
    distance = l_x_distance(suffixes1, suffixes2, chi, g3, policy)
    # Threshold for distinguishing states
    threshold = math.sqrt(2 * math.log(2 * ((len(g3)**(horizon-t + 1) - 1) // (len(g3)-1)-1) / delta) / min(len(suffixes1), len(suffixes2)))
    print(distance, threshold)
    return distance >= threshold


def l_x_distance(suffixes1, suffixes2, chi, g3, policy):
    """
    Compute the L_x distance between two sets of suffixes.

    Args:
        suffixes1: List of suffixes for the first state.
        suffixes2: List of suffixes for the second state.

    Returns:
        The Lp-infinity distance between the two sets of suffixes.
    """
    # Convert suffixes to frequency distributions
    freq1 = defaultdict(int)
    freq2 = defaultdict(int)
    arrayg3 = list(g3)
    for s in suffixes1:
        freq1[tuple(s)] += 1
    for s in suffixes2:
        freq2[tuple(s)] += 1
    print(len(g3))
    max_diff = 0
    args = [(suffixes1, freq1),(suffixes2, freq2)]
    func = partial(getprobs, arrayg3, chi)
    results = pool.map(func, args)
        
    max_diff = max([abs(p-o) for p, o in zip(results[0], results[1])])

    return max_diff

def getprobs(arrayg3, chi, suffixesfreq):
    probs = []
    for x in chi:
        prob = 0
        for s in suffixesfreq[0]:
            i = 0
            for aor in s:
                if aor in arrayg3[x[i]]:
                    i +=1
                if i == len(x):
                    prob += suffixesfreq[1][tuple(s)]/len(suffixesfreq[0])
                    break
        probs.append(prob)
    return probs



if __name__ == "__main__":
    freeze_support()
    pool = Pool(2)

    actions = [0, 1]  # Two possible actions
    observations = ['x', 'y']  # Three possible observations
    rewards = [0, 1]
    horizon = 4  # Maximum length of an episode
    delta = 0.1  # Failure probability

    # Example dataset: list of episodes
    dataset = [
        [(0,"x",0), 
        (rewards[j//8], observations[i//4 % 2], 1-rewards[j//8]), 
        (rewards[j//4 % 2], observations[i//4 % 2], 1-rewards[j//4 % 2]), 
        (rewards[j//2 % 2], observations[i % 2], 1-rewards[j//2 % 2]), 
        (rewards[j % 2], 'o0', 0)] for i in range(2*2*2) for j in range(2*2*2*2)
    ] + [
        [(0,"y",0), 
        (rewards[j//8], observations[i//4 % 2], rewards[j//8]), 
        (rewards[j//4 % 2], observations[i//2 % 2], rewards[j//4 % 2]), 
        (rewards[j//2 % 2], observations[i % 2], rewards[j//2 % 2]), 
        (rewards[j % 2], 'o0', 0)] for i in range(2*2*2) for j in range(2*2*2*2)
    ] 

    states, transitions = adact_h(dataset, delta, horizon, actions, observations, rewards, "")
    print("States:", states)
    print("Transitions:", transitions)

    # def RegORL(dataset, epsilon, delta, actions, observations, rewards, horizon, upper=0):
    #     if upper == 0:
    #         upper = (len(actions)*len(observations))**horizon * 2
    #     split = len(dataset)//2
    #     d1 = dataset[:split]
    #     d2 = dataset[split:]
    #     states, transitions = adact_h(dataset, delta/(4*len(actions)*len(observations)*upper), horizon, actions, observations, rewards)
    #     d2_new = transform(d2, transitions)
    #     policy = offlineRL(d2_new, epsilon, delta/2)

    # def offlineRL(d2, epsilon, delta):
    #     return VILCB(d2, epsilon, delta)

    # import random

    # def VILCB(data, epsilon, delta):
    #     T = math.log(len(data))/(1-epsilon)

    #     L = 2000 * math.log(2*(T+1)*S*A/epsilon)
    #     Vmax = 1/(1-epsilon)

    #     m = len(data)//(T+1)
    #     random.shuffle(data)
    #     datasets = [data[i*m:(i+1)*m] for i in range(T+1)]
    #     m0 = {} # dictionary of number of appearances for state-action pair, maybe better as 2d np array
    #     q0 = defaultdict(0) 
    #     v0 = defaultdict(0) 
    #     pi0 = {} # argmax m0 for each state
    #     for t in range(1,T+1):
    #         rt = defaultdict(0)
    #         pt = random.random() # for all state-action pairs   
    #         mt = {} # dictionary of number of appearances for state-action pair, maybe better as 2d np array
    #         bt(s,a) = Vmax * math.sqrt(L / max(mt(s,a),1))

    #         for (s, a) in [(s, a) for s in states for a in actions]:
    #             if mt(s,a) >= 1:
    #                 pt(s,a) = empirical
    #                 rt(s,a) = empirical
    #             qt(s,a) = rt(s,a) - bt(s,a) + epsilon*pt(s,a)*vt-1
    #         vt_mid = max(qt(s))
    #         pit_mid = argmax(qt(s))
    #         for s in states:
    #             if vt_mid(s) <= vt-1(s):
    #                 vt(s) = vt-1(s)
    #                 pit(s) = pit-1(s)
    #             else:
    #                 vt(s) = vt_mid(s)
    #                 pit(s) = pit_mid(s)
    #     return piT

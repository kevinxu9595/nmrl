import random

class TMaze:
    def __init__(self, length, horizon = 0, restricted = 0):
        self.length = length
        self.restricted = restricted
        self.horizon = horizon
        self.t = 0
        self.reset()

    def reset(self, horizon = None, restricted = None):
        if horizon is not None:
            self.horizon = horizon
        if restricted is not None:
            self.restricted = restricted
        self.t = 0
        self.position = 1
        self.goal_position = random.choice(['0', '1'])
        self.done = False
        return '011' if self.goal_position == '0' else '110'

    def step(self, action):
        if self.done:
            raise ValueError("Episode has finished. Please reset the environment.")
        
        if action not in ['0', '1']:
            raise ValueError("Invalid action. Valid actions are '0', '1'.")
        
        self.t += 1

        if self.position == self.length:
            if self.goal_position == action:
                reward = 4
            else:
                reward = -1
            self.done = True
            return '111', reward, self.done
        
        reward = 0

        if (self.position == 1 and action == '0') or (self.position == self.length and action == '1'):
            reward = -1
        elif action == '1':
            self.position += 1
        elif action == '0':
            self.position -= 1

        if self.position == self.length:
            observation = '010'
        else:
            observation = '101'

        if self.horizon != 0:
            if self.horizon == self.t:
                self.done = True
                observation = "111"

        return observation, reward, self.done


def generate_data(length, horizon, restricted):
    tmaze = TMaze(length=length, horizon=horizon, restricted=restricted)
    allrecords = []
    for i in range(20000):
        initial_observation = tmaze.reset()
        #print("Initial Observation:", initial_observation)
        done = False
        total = 0
        record = [('1', initial_observation, 0)]
        current_observation = initial_observation
        while not done:
            if tmaze.restricted == 2:
                if current_observation != '010':
                    action = '1'
                else:
                    action = random.choice(['0', '1'])
            elif tmaze.restricted == 1:
                if tmaze.position == 1:
                    action = '1'
                else:
                    action = random.choice(['0', '1'])
            observation, reward, done = tmaze.step(action)
            current_observation = observation
            total += reward
            #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {tmaze.position}, total: {total}")
            record.append((action, observation, reward))
        print(record)
        allrecords.append(record)

    with open('tmaze2'+str(length)+'x'+str(horizon)+'x'+str(restricted)+'.txt', 'w') as f:
        for record in allrecords:
            f.write(f"{record}\n")

def evaluate_policy(length, horizon, restricted, states, transitions, policy):
    
    tmaze = TMaze(length=length, horizon=horizon, restricted=restricted)
    allrecords = []
    average = 0
    for i in range(2000):
        initial_observation = tmaze.reset()
        #print("Initial Observation:", initial_observation)
        state = 'u0'
        done = False
        total = 0
        record = [('1', initial_observation, 0)]
        state = transitions_dict[(state, str(('1', initial_observation)))]
        t = 1
        while not done:
            action = policy_dict[(state, str(t))]
            observation, reward, done = tmaze.step(action)
            state = transitions_dict[(state, str((action, observation)))]
            total += reward
            #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {tmaze.position}, total: {total}")
            record.append((action, observation, reward))

            if t == horizon:
                done = True
            t += 1
        average += total
        print(record, average/(i+1))
        allrecords.append(record)



# Example usage
if __name__ == "__main__":
    length = 5
    horizon = 5
    restricted = 1
    # generate_data(length, horizon, restricted)

    with open('states.txt', 'r') as f:
        states = [eval(line.strip()) for line in f.readlines()]
    with open('transitions.txt', 'r') as f:
        transitions = [eval(line.strip()) for line in f.readlines()]
    with open('policy.txt', 'r') as f:
        policy = [eval(line.strip()) for line in f.readlines()]
    for state in states:
        print(state)
    print("")
    for transition in transitions:
        print(transition)
    print("")
    for p in policy:
        print(p)
    # 
    print("")
    transitions_dict = {}
    for transition in transitions:
        transitions_dict[(transition[0], transition[1])] = transition[2]
    policy_dict = {}
    for p in policy:
        policy_dict[(p[1], p[0])] = p[2]
    for t in transitions_dict:
        print(t)
    # for p in policy_dict:
    #     print(p)

    evaluate_policy(length, horizon, restricted, states, transitions_dict, policy_dict)
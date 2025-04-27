import random

class Corridor:
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
        self.position = (random.choice([0, 1]),0)
        self.done = False
        return (self.position[0], self.position[1], 0)

    def step(self, action):
        if self.done:
            raise ValueError("Episode has finished. Please reset the environment.")
        
        if action not in ['a0', 'a1']:
            raise ValueError("Invalid action. Valid actions are 'a0', 'a1'.")
        
        self.t += 1

        if self.position == self.length:
            self.done = True
        
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
                observation = "o0"

        return observation, reward, self.done


def generate_data(length, horizon, restricted):
    corridor = Corridor(length=length, horizon=horizon, restricted=restricted)
    allrecords = []
    for i in range(20000):
        initial_observation = corridor.reset()
        #print("Initial Observation:", initial_observation)
        done = False
        total = 0
        record = [('a0', initial_observation, 0)]
        current_observation = initial_observation
        while not done:
            if corridor.restricted == 2:
                if current_observation != '010':
                    action = '1'
                else:
                    action = random.choice(['0', '1'])
            elif corridor.restricted == 1:
                if corridor.position == 1:
                    action = '1'
                else:
                    action = random.choice(['0', '1'])
            observation, reward, done = corridor.step(action)
            current_observation = observation
            total += reward
            #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {corridor.position}, total: {total}")
            record.append((action, observation, reward))
        print(record)
        allrecords.append(record)

    with open('data\\corridor2'+str(length)+'x'+str(horizon)+'x'+str(restricted)+'.txt', 'w') as f:
        for record in allrecords:
            f.write(f"{record}\n")

def evaluate_policy(length, horizon, restricted):
    with open('states.txt', 'r') as f:
        states = [eval(line.strip()) for line in f.readlines()]
    with open('transitions.txt', 'r') as f:
        transitions = [eval(line.strip()) for line in f.readlines()]
    with open('policy.txt', 'r') as f:
        policy = [eval(line.strip()) for line in f.readlines()]
    transitions_dict = {}
    for transition in transitions:
        transitions_dict[(transition[0], transition[1])] = transition[2]
    policy_dict = {}
    for p in policy:
        policy_dict[(p[1], p[0])] = p[2]

    corridor = Corridor(length=length, horizon=horizon, restricted=restricted)
    allrecords = []
    average = 0
    errors = 0
    for i in range(2000):
        initial_observation = corridor.reset()
        #print("Initial Observation:", initial_observation)
        state = 'u0'
        done = False
        total = 0
        record = [('a0', initial_observation, 0)]
        state = transitions_dict[(state, str(('a0', initial_observation)))]
        t = 1
        while not done:
            try:
                action = policy_dict[(state, str(t))]
                observation, reward, done = corridor.step(action)
                state = transitions_dict[(state, str((action, observation)))]
                total += reward
                #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {corridor.position}, total: {total}")
                record.append((action, observation, reward))

                if t == horizon:
                    done = True
                t += 1
            except KeyError:
                print(f"KeyError: {state}, {t}, {action}, {record}, {total}")
                done = True
                errors += 1
        average += total
        print(record, average/(i+1))
        allrecords.append(record)
    print(errors)


# Example usage
if __name__ == "__main__":
    length = 5
    horizon = 0
    restricted = 1
    generate_data(length, horizon, restricted)

    evaluate_policy(length, horizon, restricted)
import random

class Cheese:
    def __init__(self, horizon = 9):
        self.horizon = horizon
        self.reset()
    
    def reset(self):
        self.t = 0
        self.position = random.randint(1, 10)
        self.done = False
        
        return self.get_observation()
    
    def get_observation(self):
        match self.position:
            case 1:
                return 'A'
            case _ if self.position in (2, 4):
                return 'B'
            case 3:
                return 'C'
            case 5:
                return 'D'
            case _ if self.position in (6, 7, 8):
                return 'E'
            case _ if self.position in (9, 10):
                return 'F'  

    def get_possible_actions(self, observation):
        match observation:
            case 'A':
                return ['right', 'down']
            case 'B':
                return ['left', 'right']
            case 'C':
                return ['left', 'right', 'down']
            case 'D':
                return ['left', 'down']
            case 'E':
                return ['up', 'down']
            case 'F':
                return ['up']

    def step(self, action):
        if self.done:
            raise ValueError("Episode has finished. Please reset the environment.")
        
        if action not in ['left', 'right', 'up', 'down']: 
            raise ValueError("Invalid action.")
        
        self.t += 1
        reward = 0

        match self.position:
            case 1:
                if action == 'right':
                    self.position = 2
                elif action == 'down':
                    self.position = 6
            case 2:
                if action == 'left':
                    self.position = 1
                elif action == 'right':
                    self.position = 3
            case 3:
                if action == 'left':
                    self.position = 2
                elif action == 'right':
                    self.position = 4   
                elif action == 'down':
                    self.position = 7
            case 4:
                if action == 'left':
                    self.position = 3
                elif action == 'right':
                    self.position = 5
            case 5:
                if action == 'left':
                    self.position = 4   
                elif action == 'down':
                    self.position = 8
            case 6:
                if action == 'up':
                    self.position = 1
                elif action == 'down':
                    self.position = 9
            case 7:
                if action == 'up':
                    self.position = 3
                elif action == 'down':
                    self.position = random.randint(1,10)
                    reward = 1
            case 8:
                if action == 'up':
                    self.position = 5
                elif action == 'down':
                    self.position = 10
            case 9:
                if action == 'up':
                    self.position = 6
            case 10:
                if action == 'up':
                    self.position = 8
        observation = self.get_observation()
                

        if self.horizon != 0:
            if self.horizon == self.t:
                self.done = True
                observation = "o0"

        return observation, reward, self.done


def generate_data(horizon):
    cheese = Cheese(horizon=horizon)
    allrecords = []
    for i in range(100000):
        initial_observation = cheese.reset()
        #print("Initial Observation:", initial_observation)
        done = False
        total = 0
        record = [('a0', initial_observation, 0)]
        current_observation = initial_observation
        while not done:
            actions = cheese.get_possible_actions(current_observation)
            action = random.choice(actions)
            observation, reward, done = cheese.step(action)
            current_observation = observation
            total += reward
            #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {tmaze.position}, total: {total}")
            record.append((action, observation, reward))
        print(record)
        allrecords.append(record)

    with open('data\\cheese'+'x'+str(horizon)+'.txt', 'w') as f:
        for record in allrecords:
            f.write(f"{record}\n")

def evaluate_policy(horizon):
    with open('cheese_states.txt', 'r') as f:
        states = [eval(line.strip()) for line in f.readlines()]
    with open('cheese_transitions.txt', 'r') as f:
        transitions = [eval(line.strip()) for line in f.readlines()]
    with open('cheese_policy.txt', 'r') as f:
        policy = [eval(line.strip()) for line in f.readlines()]
    transitions_dict = {}
    for transition in transitions:
        transitions_dict[(transition[0], transition[1])] = transition[2]
    policy_dict = {}
    for p in policy:
        policy_dict[(p[1], p[0])] = p[2]
    
    cheese = Cheese(horizon=horizon)
    allrecords = []
    average = 0
    errors = 0

    for i in range(2000):
        initial_observation = cheese.reset()
        #print("Initial Observation:", initial_observation)
        state = 'u0'
        done = False
        total = 0
        record = [('a0', initial_observation, 0, cheese.position)]
        state = transitions_dict[(state, str(('a0', initial_observation)))]
        t = 1
        
        while not done:
            try:
                action = policy_dict[(state, str(t))]
                observation, reward, done = cheese.step(action)
                state = transitions_dict[(state, str((action, observation)))]
                total += reward
                #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {tmaze.position}, total: {total}")
                record.append((action, observation, reward, cheese.position))

                if t == horizon:
                    done = True
                t += 1
            except KeyError:
                print(f"KeyError: {state}, {t}, {action}, {record}, {total}")
                done = True
                errors += 1
        average += total
        print(record, total, average/(i+1))
        allrecords.append(record)
    print(errors)

# Optimal score is 1.19461


# Example usage
if __name__ == "__main__":
    horizon = 6
    # generate_data(horizon)

    evaluate_policy(horizon)
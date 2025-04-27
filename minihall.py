import random

class MiniHall:
    def __init__(self, horizon = 9):
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.t = 0
        self.position = random.randint(1,12)
        self.done = False
        return self.get_observation()
    
    def get_observation(self):
        match self.position:
            case 1:
                return '0'
            case _ if self.position in (2, 8, 10):
                return '1'
            case 3:
                return '2'
            case _ if self.position in (4, 7, 9):
                return '3'
            case _ if self.position in (5, 11):
                return '4'
            case _ if self.position in (6, 12):
                return '5'  
    
    def get_possible_actions(self, observation):
        match observation:
            case '0':
                return ['left', 'right']
            case '1':
                return ['left', 'right']
            case '2':
                return ['left', 'right', 'forward']
            case '3':
                return ['left', 'right']
            case '4':
                return ['left', 'right', 'forward']
            case '5':
                return ['left', 'right', 'forward']
                

    def step(self, action):
        if self.done:
            raise ValueError("Episode has finished. Please reset the environment.")
        
        if action not in ['left', 'right', 'forward']: 
            raise ValueError("Invalid action.")
        
        self.t += 1
        reward = 0

        match self.position:
            case 1:
                if action == 'left':
                    self.position = 4
                elif action == 'right':
                    self.position = 2
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
                elif action == 'forward':
                    self.position = 7
            case 4:
                if action == 'left':
                    self.position = 3
                elif action == 'right':
                    self.position = 1
            case 5:
                if action == 'left':
                    self.position = 8
                elif action == 'right':
                    self.position = 6
                elif action == 'forward':
                    self.position = 1
            case 6:
                if action == 'left':
                    self.position = 5
                elif action == 'right':
                    self.position = 7
                elif action == 'forward':
                    self.position = 10
            case 7:
                if action == 'left':
                    self.position = 6
                elif action == 'right': 
                    self.position = 8
            case 8:
                if action == 'left':
                    self.position = 7
                elif action == 'right':
                    self.position = 5
            case 9:
                if action == 'left':
                    self.position = 12
                elif action == 'right':
                    self.position = 10
            case 10:
                if action == 'left':
                    self.position = 9
                elif action == 'right':
                    self.position = 11
            case 11:
                if action == 'left':
                    self.position = 10
                elif action == 'right':
                    self.position = 12
                elif action == 'forward':
                    self.position = random.randint(1,12)
                    reward = 1
            case 12:
                if action == 'left':
                    self.position = 11
                elif action == 'right':
                    self.position = 9
                elif action == 'forward':
                    self.position = 8
        observation = self.get_observation()
                
        if self.horizon != 0:
            if self.horizon == self.t:
                self.done = True
                observation = "o0"

        return observation, reward, self.done


def generate_data(horizon):
    minihall = MiniHall(horizon=horizon)
    allrecords = []
    for i in range(20000):
        initial_observation = minihall.reset()
        #print("Initial Observation:", initial_observation)
        done = False
        total = 0
        record = [('a0', initial_observation, 0)]
        current_observation = initial_observation
        while not done:
            actions = minihall.get_possible_actions(current_observation)
            action = random.choice(actions)
            observation, reward, done = minihall.step(action)
            current_observation = observation
            total += reward
            #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {tmaze.position}, total: {total}")
            record.append((action, observation, reward))
        print(record)
        allrecords.append(record)

    with open('data\\minihall'+'x'+str(horizon)+'.txt', 'w') as f:
        for record in allrecords:
            f.write(f"{record}\n")

def evaluate_policy(horizon):
    with open('outputs\\minihall_states.txt', 'r') as f:
        states = [eval(line.strip()) for line in f.readlines()]
    with open('outputs\\minihall_transitions.txt', 'r') as f:
        transitions = [eval(line.strip()) for line in f.readlines()]
    with open('outputs\\minihall_policy.txt', 'r') as f:
        policy = [eval(line.strip()) for line in f.readlines()]
    transitions_dict = {}
    for transition in transitions:
        transitions_dict[(transition[0], transition[1])] = transition[2]
    policy_dict = {}
    for p in policy:
        policy_dict[(p[1], p[0])] = p[2]
    
    minihall = MiniHall(horizon=horizon)
    allrecords = []
    average = 0
    errors = 0
    for i in range(2000):
        initial_observation = minihall.reset()
        #print("Initial Observation:", initial_observation)
        state = 'u0'
        done = False
        total = 0
        record = [('a0', initial_observation, 0, minihall.position)]
        state = transitions_dict[(state, str(('a0', initial_observation)))]
        t = 1
        while not done:
            try:
                action = policy_dict[(state, str(t))]
                observation, reward, done = minihall.step(action)
                state = transitions_dict[(state, str((action, observation)))]
                total += reward
                #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {tmaze.position}, total: {total}")
                record.append((action, observation, reward, minihall.position))

                if t == horizon:
                    done = True
                t += 1
            except KeyError:
                print(f"KeyError: {state}, {t}, {action}, {record}, {total}")
                done = True
                errors += 1
        average += total
        print(total, average/(i+1), i+1)
        allrecords.append(record)
    print(record)
    print("Errors:", errors)


# Example usage
if __name__ == "__main__":
    horizon = 15
    # generate_data(horizon)   

    evaluate_policy(horizon)
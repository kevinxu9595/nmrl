import random

class Cookie:
    def __init__(self, horizon = 9):
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.t = 0
        self.position = 'w'
        self.cookie = 'none'
        self.done = False
        return self.position
    
    def get_possible_actions(self, observation):
        match observation:
            case 'w':
                return ['l', 'r', 'u']
            case 'r':
                return ['d', 'p']
            case 'b':
                return ['r']
            case 'g':
                return ['l']
            case 'b_c':
                return ['r','e']
            case 'g_c':
                return ['l','e']
                

    def step(self, action):
        if self.done:
            raise ValueError("Episode has finished. Please reset the environment.")
        
        if action not in ['l', 'r', 'u', 'd', 'p', 'e']: 
            raise ValueError("Invalid action.")
        
        self.t += 1
        reward = 0

        match self.position:
            case 'w':
                if action == 'l':
                    self.position = 'b'
                elif action == 'r':
                    self.position = 'g'
                elif action == 'u':
                    self.position = 'r'
            case 'r':
                if action == 'd':
                    self.position = 'w'
                elif action == 'p':
                    self.cookie = random.choice(['b_c', 'g_c'])
            case 'b':
                if self.cookie == 'b_c' and action == 'e':
                    self.cookie = 'none'
                    reward = 1
                elif action == 'r':
                    self.position = 'w'
            case 'g':
                if self.cookie == 'g_c' and action == 'e':
                    self.cookie = 'none'
                    reward = 1
                elif action == 'l':
                    self.position = 'w'
        observation = self.position
        if self.cookie == 'b_c' and self.position == 'b':
            observation = 'b_c'
        elif self.cookie == 'g_c' and self.position == 'g':
            observation = 'g_c'
                

        if self.horizon != 0:
            if self.horizon == self.t:
                self.done = True
                observation = "o0"

        return observation, reward, self.done


def generate_data(horizon):
    cookie = Cookie(horizon=horizon)
    allrecords = []
    for i in range(20000):
        initial_observation = cookie.reset()
        #print("Initial Observation:", initial_observation)
        done = False
        total = 0
        record = [('a0', initial_observation, 0)]
        current_observation = initial_observation
        while not done:
            actions = cookie.get_possible_actions(current_observation)
            action = random.choice(actions)
            observation, reward, done = cookie.step(action)
            current_observation = observation
            total += reward
            #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {tmaze.position}, total: {total}")
            record.append((action, observation, reward))
        print(record)
        allrecords.append(record)

    with open('data\\cookie'+'x'+str(horizon)+'.txt', 'w') as f:
        for record in allrecords:
            f.write(f"{record}\n")

def evaluate_policy(horizon):
    with open('outputs\\cookie_states.txt', 'r') as f:
        states = [eval(line.strip()) for line in f.readlines()]
    with open('outputs\\cookie_transitions.txt', 'r') as f:
        transitions = [eval(line.strip()) for line in f.readlines()]
    with open('outputs\\cookie_policy.txt', 'r') as f:
        policy = [eval(line.strip()) for line in f.readlines()]
    transitions_dict = {}
    for transition in transitions:
        transitions_dict[(transition[0], transition[1])] = transition[2]
    policy_dict = {}
    for p in policy:
        policy_dict[(p[1], p[0])] = p[2]
    
    cookie = Cookie(horizon=horizon)
    allrecords = []
    average = 0
    errors = 0
    for i in range(2000):
        initial_observation = cookie.reset()
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
                observation, reward, done = cookie.step(action)
                state = transitions_dict[(state, str((action, observation)))]
                total += reward
                #print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {tmaze.position}, total: {total}")
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
    horizon = 9
    # generate_data(horizon)

    evaluate_policy(horizon)
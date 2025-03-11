import random

class TMaze:
    def __init__(self, length, horizon = None, restricted = None):
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
        self.goal_position = random.choice(['North', 'South'])
        self.done = False
        return '011' if self.goal_position == 'North' else '110'

    def step(self, action):
        if self.done:
            raise ValueError("Episode has finished. Please reset the environment.")
        
        if action not in ['North', 'South', 'East', 'West']:
            raise ValueError("Invalid action. Valid actions are 'North', 'South', 'East', 'West'.")
        
        self.t += 1

        if self.position == self.length and action in ['North', 'South']:
            if self.goal_position == action:
                reward = 4
            else:
                reward = -1
            self.done = True
            return '111', reward, self.done
        
        reward = 0

        if action in ['North', 'South'] or (self.position == 1 and action == 'West') or (self.position == self.length and action == 'East'):
            reward = -1
        elif action == 'East':
            self.position += 1
        elif action == 'West':
            self.position -= 1

        if self.position == self.length:
            observation = '010'
        else:
            observation = '101'

        if self.horizon is not None:
            if self.horizon == self.t:
                self.done = True
                observation = "111"

        return observation, reward, self.done

# Example usage
if __name__ == "__main__":
    length = 5
    horizon = 5
    restricted = 2
    tmaze = TMaze(length=length, horizon=horizon, restricted=restricted)
    allrecords = []
    for i in range(100):
        initial_observation = tmaze.reset()
        print("Initial Observation:", initial_observation)
        done = False
        total = 0
        record = [('East', initial_observation, 0)]
        current_observation = initial_observation
        while not done:
            if tmaze.restricted == 2:
                if current_observation != '010':
                    action = 'East'
                else:
                    action = random.choice(['North', 'South'])
            elif tmaze.restricted == 1:
                if current_observation != '010':
                    action = random.choice(['East', 'West'])
                else:
                    action = random.choice(['North', 'South'])
            else:
                action = random.choice(['North', 'South', 'East', 'West'])
            observation, reward, done = tmaze.step(action)
            current_observation = observation
            total += reward
            print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, position: {tmaze.position}, total: {total}")
            record.append((action, observation, reward))
        print(record)
        allrecords.append(record)

    with open('tmaze'+str(length)+'x'+str(horizon)+'x'+str(restricted)+'.txt', 'w') as f:
        for record in allrecords:
            f.write(f"{record}\n")
def RegORL(dataset, epsilon, delta, actions, observations, rewards, horizon, upper=0):
        if upper == 0:
            upper = (len(actions)*len(observations))**horizon * 2
        random.shuffle(dataset)
        split = len(dataset)//2
        d1 = dataset[:split]
        d2 = dataset[split:]
        states, transitions = adact_h(dataset, delta/(4*len(actions)*len(observations)*upper), horizon, actions, observations, rewards)
        d2_new = transform(d2, transitions)
        policy = offlineRL(d2_new, epsilon, delta/2)

    def offlineRL(d2, epsilon, delta):
        return subVILCB(d2, epsilon, delta)

    import random

    def subVILCB(data, epsilon, delta):
        random.shuffle(data)
        split = len(data)//2
        d1 = data[:split]
        d2 = data[split:]
        nmain(s)
        naux(s)

        for s in states:
            for h in range(horizon):
                ntrim(s) = max(naux(s)- 10*math.sqrt(naux(s)*math.log(horizon*len(states)/delta)), 0)
        
        d(0)= d(trim)
        return VILCB(d(0), epsilon, delta)
    
    def VILCB(data, epsilon, delta):
        V= [0]
        for h in range(horizon):
            compute the empirical transition kernel Pbh according to (51).
        for s in states:
            for a in actions:
                compute the penalty term bh(s, a) according to (55).
        Qbh(s, a) = max(rh(s, a) + Pbh,s,aVbh+1 − bh(s, a), 0)
        for s in states:
            Vbh(s) = maxa (Qbh(s, a))
            πbh(s) = arg maxa (Qbh(s, a))
        return πb = {πbh}1≤h≤H.
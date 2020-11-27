import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
gamma = 0.99
w = np.matrix([0,0,0,0,0,0,0,0,0,0,0,0])
actions = [-40, 40]

def log(log_message):
    """

    DESCRIPTION:
    - Adds a log message "log_message" to a log file.

    """

    # open the log file and make sure that it's closed properly at the end of the
    # block, even if an exception occurs:
    with open("/home/felipe/Documentos/abet/openai/src/sdip/log_batch.log", "a") as log_file:
        # write the log message to logfile:
        log_file.write(log_message)
        log_file.write("\n") # (so the next message is put on a new line)

def normalize_angle(angle):
    """
    3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
    from the closest multiple of 2*pi)
    """
    normalized_angle = abs(angle)
    normalized_angle = normalized_angle % (2*np.pi)
    if normalized_angle > np.pi:
        normalized_angle = normalized_angle - 2*np.pi
    normalized_angle = abs(normalized_angle)
    return normalized_angle

def X(state, action):
    x = state.item(0)
    theta = state.item(1)
    phi = state.item(2)
    x_dot = state.item(3)
    theta_dot = state.item(4)
    phi_dot = state.item(5)
    X = np.matrix([[normalize_angle(theta)], [theta % (2*np.pi)],
            [normalize_angle(phi)], [phi % (2*np.pi)],
            [abs(theta_dot)], [theta_dot],
            [abs(phi_dot)], [phi_dot],
            [action*theta], [action*phi],
            [action*theta_dot], [action*phi_dot]])
    return X

def q_hat(state, action, w):
    X_value = X(state, action)
    output = np.array(w*X_value)[0][0]
    return output

def pi(state, w):
    qs = []
    for action in actions:
        qs.append(q_hat(state, action, w))
    max_index = np.argmax(qs)
    action = actions[max_index]
    return action

# generate experience data:
D = {}
index = 0
while index < 99:
    S = env.reset()
    for t in range(100):
        #env.render()
        A = actions[int(round(np.random.uniform(0,1)))]
        S_, R, done, info = env.step(A)
        D[index] = {}
        D[index]["S"] = S
        D[index]["A"] = A
        D[index]["R"] = R
        D[index]["S_"] = S_
        index += 1
        S = S_

        if done:
            print("Episode finished after " + str(t+1) + " timesteps")
            break

timesteps = []
index = 0


while True:
    S = env.reset()
    for t in range(100000):
        #env.render()
        A = pi(S, w)
        #print(A)
        S_, R, done, info = env.step(A)
        D[index] = {}
        D[index]["S"] = S
        D[index]["A"] = A
        D[index]["R"] = R
        D[index]["S_"] = S_
        if index >= 99:
            index = 0
        else:
            index += 1

        # update w:
        sum_inv = 0
        sum = 0
        for t2 in range(100):
            St = D[t2]["S"]
            At = D[t2]["A"]
            Rt_1 = D[t2]["R"]
            St_1 = D[t2]["S_"]
            sum_inv += X(St,At)*np.transpose((X(St,At)-gamma*X(St_1,pi(St_1,w))))
            sum += X(St,At)*Rt_1
        w = np.linalg.pinv(sum_inv)*sum
        w = np.transpose(w)

        #print(w)
        #print(reward)
        S = S_

        if done:
            #print("Episode finished after " + str(t+1) + " timesteps")
            timesteps += [t+1]
            log(str(t+1))
            log(str(w))
            break
plt.plot(timesteps)
plt.show()

#[-6.46, -7.89, -8.74, -30.85, -30.07, -5.055,- 33.34, 23.09, 72.94, -38.84, 1.77, -55.54]
#[0.67428356, -0.06215373, 0.71389644, -0.04985835, -0.35220665, 0.91792232, 0.03106055, 0.78030967, 0.12260051, -0.18088769, 0.00512933, -0.07230826]

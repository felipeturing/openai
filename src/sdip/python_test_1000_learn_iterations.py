import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
gamma = 0.99
alpha = 0.0001
action = 1
# setting w_0
#w = np.array([-3.26235474e+08,1.19486826e+07,-6.25423208e+08,8.44361669e+08,4.19343699e+08,-6.45561216e+07,-2.27575813e+09,7.80709627e+08, 6.43126956e+09,-8.54103946e+09,1.05744777e+08,-5.98016330e+08])
#w = np.array([-6.46, -7.89, -8.74, -30.85, -30.07, -5.055,- 33.34, 23.09, 72.94, -38.84, 1.77, -55.54])
w = np.array([-3.26235474e+08,1.19486826e+07,-6.25423208e+08,8.44361669e+08,4.19343699e+08,-6.45561216e+07,-2.27575813e+09,7.80709627e+08, 6.43126956e+09,-8.54103946e+09,1.05744777e+08,-5.98016330e+08])
#w =np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
delta_w = np.array([0,0,0,0,0,0,0,0,0,0,0])

def log(log_message):
    """

    DESCRIPTION:
    - Adds a log message "log_message" to a log file.

    """

    # open the log file and make sure that it's closed properly at the end of the
    # block, even if an exception occurs:
    with open("/home/turing/Repos/openai/src/sdip/log.xor", "a") as log_file:
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

def state_action_to_features(state, action):
    x = state.item(0)
    theta = state.item(1)
    phi = state.item(2)
    x_dot = state.item(3)
    theta_dot = state.item(4)
    phi_dot = state.item(5)
    X = np.array([normalize_angle(theta), theta % (2*np.pi),
            normalize_angle(phi), phi % (2*np.pi),
            abs(theta_dot), theta_dot,
            abs(phi_dot), phi_dot,
            action*theta, action*phi,
            action*theta_dot, action*phi_dot])
    return X

def q_hat(state, action, w):
    X = state_action_to_features(state, action)
    output = np.dot(X,w)
    return output

# epislon-greedy:
def get_action(state, w):
    actions = [-40, 40]
    qs = []
    for action in actions:
        qs.append(q_hat(state, action, w))
    max_index = np.argmax(qs) #el indice de la accion que hace que qs se maximize
    action = actions[max_index] #con el indice ya podemos encontrar la accion que maximiza qs
    return action

timesteps = []
for i_episode in range(160):
    #empezamos un nuevo episodio
    state = env.reset() #inicializamos la acción s_0 con la posicion del primer pendulo alejado muy poco del equilibrio
    action = get_action(state, w)
    for t in range(100000): #vamos t=0 hasta t=100000
        env.render()
        action = get_action(state, w) # encontrar la accion que maximiza los valores de qs para el estado actuals
        observation, reward, done, info = env.step(action) # con esa accion encontrada obtenemos el estado siguiente(observation), la recompensa y done que es verdadero si el estado siguiente no es un estado de equilibrio(estado objetivo).
        # update w
        delta_w = (alpha*(reward + gamma*q_hat(observation, get_action(observation, w), w) - q_hat(state, action, w)))*state_action_to_features(state, action)
        w = np.add(w,delta_w)
        state = observation # el estado siguiente ahora es el estado actual (t <- t + 1)


        #si done es verdadero entonces el estado siguiente no es un estado objetivo, luego termina este episodio y debemos comenzar otro, pero ahora empezando ya no con los valores iniciales sino con los ya actualizados, es decir recordando por donde no sé tiene que ir o sí se tiene que ir.
        if done:
            print("Episodio finalizado luego de " + str(t+1) + " iteraciones")
            timesteps += [t+1]
            print(w)
            break
plt.plot(timesteps)
log(str(timesteps))
log(str(w))
plt.show()

#[-6.46, -7.89, -8.74, -30.85, -30.07, -5.055,- 33.34, 23.09, 72.94, -38.84, 1.77, -55.54]
#[-3.26235474e+08,1.19486826e+07,-6.25423208e+08,8.44361669e+08,4.19343699e+08,-6.45561216e+07,-2.27575813e+09,7.80709627e+08, 6.43126956e+09,-8.54103946e+09,1.05744777e+08,-5.98016330e+08]


# Ruta del archivo cartpole-v1: ./.local/lib/python3.8/site-packages/gym/envs/classic_control/cartpole.py

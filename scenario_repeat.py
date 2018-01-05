import numpy as np
from matplotlib import pyplot as plt

# Friis Equation Pr = (PtGtGr)(c^2)/((4piRf)^2)
# PtGt is assumed to be 20dBm which is maximum legal value for 2.4GHz Pint-to_MultiPoint set by European Telecommunications Standards Institute
# Gr is assumed to be 0dBi
# c/f is -9dB where c is speed light and f is assumed to be 2.4GHz for 1 m distance
# 1/(4pi) is -11dB
# dBm(Pr) = dBm(PtGt) + dBi(Gr) + 2*dB(c/f) + 2*dB(1/(4pi)) + 2*dB(1/R) where R is in meters unit
# dBm(Pr) = 20dBm - 18dB - 22dB + 20*log(1/R) = -20 dBm + 20*log(1/R) dB
def distanceToSS(distance):
    return -20 * np.log10(distance) - 20.0

def ssToDistance(signal_strength):
    return 10 ** ((signal_strength + 20.0) / (-20))

def pointToDistance(routers,client):
    diff = routers[:,:] - client[:,None]
    return np.sqrt(np.sum(diff**2,axis=0))

def plotInit(routers, client, area_edge, signals):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(routers[0], routers[1], s=20, c='b', label='routers')
    ax.scatter(client[0], client[1], s=40, c='y', label='client')
    ax.set_xlim(0, area_edge)
    ax.set_ylim(0, area_edge)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    # ax.legend()

    for i, txt in enumerate(signals):
        ax.annotate(round(txt, 2), (routers[0][i], routers[1][i] + 0.1))

    return ax


area_edge = 40 # One edge size in meters
number_routers = 8 # number of WiFi Routers

differences = list()
for k in range(100):
    routers = np.random.rand(2, number_routers) * area_edge # Area is 10m x 10m
    client = np.random.rand(2) * area_edge / 4 + area_edge * (3./8) # Places client near to senter
    distances = pointToDistance(routers, client)
    signals = distanceToSS(distances)

    ITERATIONS = 10000
    POSITION_CHANGE = area_edge / 10
    BETA_CHANGE = 1.01
    cost_multiplier = 0.1

    nearest_router = np.argmin(distances)
    client_estimation = np.copy(routers[:,nearest_router])
    distances_estimation = pointToDistance(routers, client_estimation)
    distances_delta = np.sum((distances_estimation - distances) ** 2)

    client_history = list()
    history_constant = 10
    count = 0

    difference_record = list()
    for index in range(ITERATIONS):
        client_estimation_iter = np.clip(np.copy(client_estimation) + (np.random.rand(2) -0.5) * POSITION_CHANGE, 0, area_edge)
        distances_estimation_iter = pointToDistance(routers, client_estimation_iter)
        distances_delta_iter = np.sum((distances_estimation_iter - distances) ** 2)
        delta_cost = (distances_delta_iter - distances_delta) * cost_multiplier
        if np.random.rand() < np.exp(-delta_cost):
            client_estimation = client_estimation_iter
            distances_estimation = distances_estimation_iter
            distances_delta = distances_delta_iter
            count += 1
        cost_multiplier *= BETA_CHANGE
        difference_record.append(np.sum((client_estimation - client)**2))
    differences.append(difference_record)

fig, ax = plt.subplots(1, 1)
for single_diff in differences:
    plt.semilogy(single_diff, c=(0,0,1,0.5))

plt.xlabel('iterations')
plt.ylabel('delta')
plt.show()
plt.pause(0.5)
pass
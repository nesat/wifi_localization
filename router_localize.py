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

def pointToDistance3D(routers,client):
    diff = routers[:,:, None] - client[:,None, :]
    return np.sqrt(np.sum(diff**2,axis=0))

def pointToDistance2D(routers,client):
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
        ax.annotate(round(txt[0], 2), (routers[0][i], routers[1][i] + 0.1))
        ax.annotate(i, (routers[0][i] + 0.3, routers[1][i] - 1.3))

    return ax


area_edge = 40 # One edge size in meters
number_routers = 8 # number of WiFi Routers
routers = np.random.rand(2, number_routers) * area_edge # Area is 10m x 10m
clients = np.random.rand(2) * area_edge / 4 + area_edge * (3./8) # Places client near to senter
clients = np.asarray([clients[0]*np.ones(4) + [0,1,1,0], clients[1]*np.ones(4) + [0,0,1,1]])
distances = pointToDistance3D(routers, clients)
signals = distanceToSS(distances)


ax = plotInit(routers, clients, area_edge, signals)
plt.show()

ITERATIONS = 10000
POSITION_CHANGE = area_edge / 10
BETA_CHANGE = 1.01

router_estimations = list()

for router_index in range(number_routers):
    cost_multiplier = 0.1
    router_estimation = np.random.rand(2)
    distances_estimation = pointToDistance2D(clients, router_estimation)
    distances_delta = np.sum((distances_estimation - distances[router_index, :]) ** 2)
    count = 0

    for index in range(ITERATIONS):
        router_estimation_iter = np.clip(np.copy(router_estimation) + (np.random.rand(2) -0.5) * POSITION_CHANGE, 0, area_edge)
        distances_estimation_iter = pointToDistance2D(clients, router_estimation_iter)
        distances_delta_iter = np.sum((distances_estimation_iter - distances[router_index, :]) ** 2)
        delta_cost = (distances_delta_iter - distances_delta) * cost_multiplier
        if np.random.rand() < np.exp(-delta_cost):
            router_estimation = router_estimation_iter
            distances_estimation = distances_estimation_iter
            distances_delta = distances_delta_iter
            count += 1
        cost_multiplier *= BETA_CHANGE
    router_estimations.append(router_estimation)

print count
ax = plotInit(routers, clients, area_edge, signals)
for i, location in enumerate(router_estimations):
    ax.scatter(location[0], location[1], s=40, c=(1, 0, 0, 0.5), label='router')
    ax.annotate(i, (location[0], location[1] + 0.1))
plt.show()

pass
import os
import random
import json
import pandas as pd
import numpy as np
import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3 import PPO, SAC, A2C

import matplotlib.pyplot as plt 

SEED = 100
np.random.seed(SEED)

materials = ["I", "AIP", "AIS_As", "AIS_Am", "PA", "PC", "TEE"]
products = ["PA", "PC", "TEE"]
secondary_sites = ["Asia", "America"]
resources = ["Granulates", "Compress", "Coat", "QC", "Packing"]
# intermediates = ['TA', 'TB', 'I1', 'I2', 'I3', 'I4']
T_set = np.arange(1, 25).tolist()


# F = {("PA", t): (t - 1) * 4000 / 100 + 5000 for t in T_set}
# F_ = {("PC", t): (t - 1) * 1e4 / 100 + 10000 for t in T_set}
# F__ = {("TEE", t): (t - 1) * 1e5 / 100 + 400000 for t in T_set}
F = {("PA", t): 5000*(1+.1*int(t>=5) - .2*int(t>=10) + .3*int(t>=15) - .4*int(t>=20)) for t in T_set}
F_ = {("PC", t): 10000*(1+.1*int(t>=5) - .2*int(t>=10) + .3*int(t>=15) - .4*int(t>=20)) for t in T_set}
F__ = {("TEE", t): 300000*(1+.1*int(t>=5) - .2*int(t>=10) + .3*int(t>=15) - .4*int(t>=20)) for t in T_set}
F.update(F_)
F.update(F__)

config = {
        "M": {None: materials},
        "P": {None: products},
        "L": {None: secondary_sites},
        "R": {None: resources},
        "T": {None: T_set},
        # "demand_type": {None: 'deterministic'},
        "demand_type": {None: 'random'},
        "backoff": {None: 0.4},
        "rho": {None: 1e6},
        "rho_inc": {None: 1},
        # Initial storage
        "S0": {"PA": 15e3, "PC": 6e4, "TEE": 2e6}, 
        "SAIP0": {None: 3000},
        "SI0": {None: 3000},
        "Plan": {None: 5}, # not sure what this is
        "SAIS0": {"Asia": 480, "America": 360}, # previously 200
        # 'ITAstar': {None: 25e4}, 'ITBstar': {None: 25e4},
        #Safety storage
        "IAIPstar0": {None: 3000},
        "IIstar0": {None: 3000},
        "Istar0": {"PA": 15e3, "PC": 6e4, "TEE": 2e6},
        "IAISstar0": {"Asia": 400, "America": 300},
        #Costs
        "CT": {"Asia": 15, "America": 10}, #Transport
        "CS": {"PA": 0.1, "PC": 0.15, "TEE": 0.1}, # Storage
        "CS_SAIS": {"Asia": 0.02, "America": 0.03},
        # 'CS_SAIS': {'Asia': 0, 'America': 0},
        "CS_AIP": {None: 0.02},
        "CS_I": {None: 0.01},
        # 'CS_AIP': {None: 0}, 'CS_I': {None: 0},
        "RM_Cost": {None: 0.1}, #Raw material cost
        "AI_Cost": {None: 0.5}, # AI cost (should not matter for the whole company)
        # 'RM_Cost': {None: 0}, 'AI_Cost': {None: 0},
        "Price": {"PA": 10, "PC": 10, "TEE": 1},
        "CP": {"PA": 1.2, "PC": 1.2, "TEE": 0.06}, # Production
        "CP_I": {None: 0.05e-1},
        "CP_AI": {None: 0.2e-1},
        # 'CP_I': {None: 0}, 'CP_AI': {None: 0},
        "SP_AI": {None: 2500}, # Selling price
        # 'SP_AI': {None: 0},
        "LT": {"Asia": 4, "America": 3}, # lead time
        "Q": {"PA": 20e-6 * 12 / 0.02, "PC": 6e-4 / 0.02, "TEE": 3e-5}, # AI to material conversion
        # No FPAI(T), FPTA(T), FPTB(T), FPI4(T), FPI3(T), FPI2(T), FPI1(T), FTP(T,K)
        "A": {  # Resource availability
            ("Asia", "Granulates"): 120,
            ("Asia", "Compress"): 480,
            ("Asia", "Coat"): 480,
            ("Asia", "QC"): 1800,
            ("Asia", "Packing"): 320,
            ("America", "Granulates"): 120,
            ("America", "Compress"): 120,
            ("America", "Coat"): 120,
            ("America", "QC"): 720,
            ("America", "Packing"): 160,
        },
        "U": { # Resource time utilised per product
            ("Asia", "Granulates"): 133e-3,
            ("Asia", "Compress"): 350e-3,
            ("Asia", "Coat"): 333e-3,
            ("Asia", "QC"): 2,
            ("Asia", "Packing"): 267e-3,
            ("America", "Granulates"): 133e-3,
            ("America", "Compress"): 350e-3,
            ("America", "Coat"): 333e-3,
            ("America", "QC"): 2,
            ("America", "Packing"): 267e-3,
        },
        "X": { # matches between final products and secondary sites
            ("Asia", "PA"): 1,
            ("Asia", "PC"): 0,
            ("Asia", "TEE"): 1,
            ("America", "PA"): 0,
            ("America", "PC"): 1,
            ("America", "TEE"): 0,
        },
        "F": F, # Demand forecast
}

### CONSTANTS ###
# MAX_ACCOUNT_BALANCE = 2147483647

class SC_base(gym.Env):
    """An inventory management problem for a 3-node supply chain"""

    def __init__(self, config):

        self.data = config.copy()
        self.materials = config['M'][None]
        self.products = config['P'][None]
        self.locations = config['L'][None]
        self.resources = config['R'][None]
        self.timesteps = int(config['T'][None][-1])

        self.time = 1
        self.reward = 0
        self.penalty = 0
        self.sales = 0
        self.cumulative_reward = 0
        self.cumulative_sales = 0
        self.cumulative_penalty = 0
        self.cumulative_res_viol = 0
        self.cumulative_istar_constr = 0

        self.backoff = config['backoff'][None]
        self.penalty_cost = config['rho'][None]
        self.penalty_inc = config['rho_inc'][None]

        self.lead_time = config['LT']
        self.Q = config['Q']
        self.res_avail = config['A']
        self.time_per_res = config['U']
        self.product_at_loc = config['X']
        self.forecast = config['F']
        self.demand_type = config['demand_type'][None]

        self.transport_buffer = {
            "Asia": [0]*self.lead_time["Asia"],
            "America": [0]*self.lead_time["America"],
        }

        self.safe_storage = {
            "I": config['IIstar0'][None],
            "AIP": config['IAIPstar0'][None],
            "AIS_As": config['IAISstar0']['Asia'],
            "AIS_Am": config['IAISstar0']['America'],
            "PA": config['Istar0']['PA'],
            "PC": config['Istar0']['PC'],
            "TEE": config['Istar0']['TEE'],
        }

        self.cost_production = {
            "I": config['CP_I'][None],
            "AIP": config['CP_AI'][None],
            "AIS_As": config["CT"]["Asia"],
            "AIS_Am": config["CT"]["America"],
            "PA":  config['CP']['PA'],
            "PC":  config['CP']['PC'],
            "TEE": config['CP']['TEE'],
        }

        self.cost_store = {
            "I": config['CS_I'][None],
            "AIP": config['CS_AIP'][None],
            "AIS_As": config['CS_SAIS']['Asia'],
            "AIS_Am": config['CS_SAIS']['America'],
            "PA":  config['CS']['PA'],
            "PC":  config['CS']['PC'],
            "TEE": config['CS']['TEE'],
        }

        self.price = {
            "PA":  config['Price']['PA'],
            "PC":  config['Price']['PC'],
            "TEE": config['Price']['TEE'],
            "AI": config['SP_AI'][None]
        }

        self.init_states = np.array([
            config['SI0'][None], # inventory
            config['SAIP0'][None],
            config['SAIS0']['Asia'],
            config['SAIS0']['America'],
            config['S0']['PA'],
            config['S0']['PC'],
            config['S0']['TEE'],
            config['F']['PA',1], # forecast
            config['F']['PC',1],
            config['F']['TEE',1],
            ])

        ### Initial states
        self.states = self.init_states
        self.states_vector = [list(self.states)]
        self.demand_vector = []

        self.upper_states = np.array([self.safe_storage[self.materials[i]]*10 for i in range(len(self.materials))] + [10e3, 20e3, 600e3])
        dummy = np.array([1e4, 1e4, 1e3, 1e3, 1e5, 1e6, 1e7, 1e5, 1e5, 1e7])*10
        for i in range(len(self.upper_states)):
            self.upper_states[i] = min(self.upper_states[i], dummy[i])
        self.states_scaled = self._scale(self.states, self.upper_states)
        
        self.upper_action = self.upper_states.copy()
        # dummy = np.array([1e4, 1e4, 1e3, 1e3, 1e5, 1e5, 5e5, 1e5, 1e5, 5e5])
        for i in range(len(self.upper_states)):
            self.upper_action[i] = min(self.upper_action[i], 5e5)
        self.actions_vector = []
        

        # Actions: [P_I, P_AIP, TP_As, TP_Am, P_PA, P_PC, P_TEE, F_PA, F_PC, F_TEE]
        self.action_space = spaces.Box(
            low=  -np.ones(len(self.states)), 
            high= np.ones(len(self.states)),
            )

        # Inventory of all materials (7x) + forecast of products (3x)
        self.observation_space = spaces.Box(
            low=  -np.ones(len(self.states)), 
            high= np.ones(len(self.states)), 
            shape=(10, ),
            )

    def _demand_realisation(self):
        ### Double-check
        ###

        if self.demand_type=='deterministic':
            D = self.states[-3:]
        else:
            D = np.array([    
                    np.random.uniform(
                        0.8*self.states[-3+p], 1.2*self.states[-3+p]
                    ) for p in range(len(self.products))
                ])
        return D


    def _take_action(self, action, demand):

        self.states[0] = max(0, 
            self.states[0] + action[0] - 1.1 * action[1]
        )
        self.states[1] = max(0, 
            self.states[1] + action[1] - (action[2]+action[3])
        )
        self.states[2] = max(0,
            self.states[2] + self.transport_buffer["Asia"][0] - 1.1*(
                action[4] * self.Q["PA"] + action[6] * self.Q["TEE"] 
            )
        )
        self.states[3] = max(0,
            self.states[3] + self.transport_buffer["America"][0] - 1.1*(
                action[5] * self.Q["PC"] 
            )
        )
        self.dummy_sales = [
            min(demand[0],action[7], self.states[4] + action[4]),
            min(demand[1],action[8], self.states[5] + action[5]),
            min(demand[2],action[9], self.states[5] + action[6])
        ]
        self.states[4] = max(0, 
            self.states[4] + action[4] - self.dummy_sales[0]
        )
        self.states[5] = max(0, 
            self.states[5] + action[5] - self.dummy_sales[1]
        )
        self.states[6] = max(0, 
            self.states[6] + action[6] - self.dummy_sales[2]
        )

        self.transport_buffer["Asia"] = self.transport_buffer["Asia"][1:] + [action[2]]
        self.transport_buffer["America"] = self.transport_buffer["America"][1:] + [action[3]]

        if self.time < self.timesteps:
            self.states[7] = self.forecast["PA", self.time+1]
            self.states[8] = self.forecast["PC", self.time+1]
            self.states[9] = self.forecast["TEE", self.time+1]

        assert (self.states >= -1e-5).all()
    
    def _get_reward(self, action, demand):
        
        M_len = len(self.materials)
        P_len = len(self.products)

        prod_cost = sum(
            [self.cost_production[self.materials[i]]*action[i] for i in range(M_len)]
        )
        
        store_cost = sum(
            [self.cost_store[self.materials[i]]*action[i] for i in range(M_len)]
        )

        sales = sum(
            [self.price[self.products[i]]*self.dummy_sales[i] for i in range(P_len)]
        )

        res_viol_backoff = sum(
            max(0, 
                self.time_per_res[l,r]* sum(
                    action[-6+i]*self.product_at_loc[l, self.products[i]]*self.Q[self.products[i]] for i in range(P_len)
                )/self.res_avail[l, r] - (1-self.backoff)
            )**2 for l in self.locations for r in self.resources
        ) # / len(self.locations) / len(self.resources)

        prod_UL = sum(
            max(0, action[-6+i] - 500e3)**2 for i in range(P_len)
        ) + max(0, action[0] - 500e3)**2 + max(0, action[1] - 500e3)**2
        
        if self.time > 4:
            istar_constr_backoff = sum(
                [max((1-self.backoff) - self.states[i]/self.safe_storage[materials[i]], 0)**2 for i in range(M_len)]
            )
        else:
            istar_constr_backoff = sum(
                [max((1-self.backoff)/4 - self.states[i]/self.safe_storage[materials[i]], 0)**2 for i in range(M_len)]
            )


        res_viol = sum(
            max(0, 
                self.time_per_res[l,r]* sum(
                    action[-6+i]*self.product_at_loc[l, self.products[i]]*self.Q[self.products[i]] for i in range(P_len)
                )/self.res_avail[l, r] - 1
            )**2 for l in self.locations for r in self.resources
        ) # / len(self.locations) / len(self.resources)

        if self.time > 4:
            istar_constr = sum(
                [max(1 - self.states[i]/self.safe_storage[materials[i]], 0)**2 for i in range(M_len)]
            )
        else:
            istar_constr = sum(
                [max(1/4 - self.states[i]/self.safe_storage[materials[i]], 0)**2 for i in range(M_len)]
            )


        self.penalty = (res_viol_backoff + prod_UL + istar_constr_backoff)
        # dummy1 = min(self.penalty, 100000)
        # dummy2 = max(0, self.penalty - 100000)
        dummy1 = min(self.penalty, 10)
        dummy2 = max(0, self.penalty - 10)
        self.penalty = dummy1 + dummy2/1e2


        self.cumulative_penalty += self.penalty
        self.cumulative_res_viol += res_viol
        self.cumulative_istar_constr += istar_constr
        self.reward = (sales - prod_cost - store_cost - self.penalty*self.penalty_cost)/1e3
        self.sales = sales/1e3
        self.cumulative_sales += self.sales
        self.cumulative_reward += self.reward

        ###
        self.penalty_cost *= self.penalty_inc
        ###

        # if self.cumulative_reward < -1e3:
        #     print('what')

        return self.reward

    def _descale(self, scaled, upper_bound):
        return (scaled + 1)/2*upper_bound
    
    def _scale(self, unscaled, upper_bound):
        return unscaled/upper_bound*2 - 1

    def _restrict_action(self, action):
        for i in range(len(self.materials)):
            action[i] = max(min(action[i], self.upper_states[i] - self.states[i]),0)
        return action

    def step(self, action_scaled):
        # Execute one time step within the environment
        demand = self._demand_realisation()
        self.demand_vector += [list(demand)]
        action = self._descale(action_scaled, self.upper_action)
        action = self._restrict_action(action)
        self._take_action(action, demand)

        self.time += 1

        done = (self.time >= self.timesteps) #or (self.cumulative_penalty > 5e7)

        self._get_reward(action, demand)
        
        self.states_scaled = self._scale(self.states, self.upper_states)

        self.actions_vector += [list(action)]
        self.states_vector += [list(self.states)]

        return self.states_scaled, self.reward, done, {}

    def reset(self):
        
        self.states = self.init_states
        self.states_vector = [list(self.states)]
        self.actions_vector = []
        self.demand_vector = []
        self.states_scaled = self._scale(self.states, self.upper_states)
        self.time = 1
        self.reward = 0
        self.penalty = 0
        self.cumulative_reward = 0
        self.cumulative_penalty = 0
        self.cumulative_res_viol = 0
        self.cumulative_istar_constr = 0
        
        self.sales = 0
        self.cumulative_sales = 0

        return self.states_scaled

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.time}')
        print(f'Step Profit: {self.reward}')
        print(f'Total Profit: {self.cumulative_reward}')
        print(f'Step  Sales: {self.sales}')
        print(f'Total Sales: {self.cumulative_sales}')
        print(f'Total Penalty: {self.cumulative_penalty*self.penalty_cost}')
        print(f'Total Resource Penalty: {self.cumulative_res_viol*self.penalty_cost}')
        print(f'Total Storage Penalty: {self.cumulative_istar_constr*self.penalty_cost}')

    def plots(self):

        fig, axs = plt.subplots(5, 2, figsize=(15, 20))
        
        S_PC = np.array(self.states_vector)[:,-5]
        S_PA = np.array(self.states_vector)[:,-6]
        S_TEE = np.array(self.states_vector)[:,-4]

        P_PC =  np.array(self.actions_vector)[:,-5]
        F_PC =  np.array(self.states_vector)[:,-2]
        SA_PC = np.array(self.actions_vector)[:,-2]

        SAIS_Am = np.array(self.states_vector)[:, 3]
        SAIS_As = np.array(self.states_vector)[:, 2]

        P_PA =   np.array(self.actions_vector)[:,-6]
        P_TEE =  np.array(self.actions_vector)[:,-4]
        SA_PA =  np.array(self.actions_vector)[:,-3]
        SA_TEE = np.array(self.actions_vector)[:,-1]
        F_PA =   np.array(self.states_vector)[:,-3]
        F_TEE =  np.array(self.states_vector)[:,-1]

        TP_Am = np.array(self.actions_vector)[:, 3]
        TP_As = np.array(self.actions_vector)[:, 2]

        PI =  np.array(self.actions_vector)[:, 0]
        PAI = np.array(self.actions_vector)[:, 1]
        SI =    np.array(self.states_vector)[:, 0]
        SAIP =  np.array(self.states_vector)[:, 1]

        N_s = len(S_PC)
        N_a = N_s-1

        axs[0,0].plot(np.arange(N_a), PI, label='I production')
        axs[0,0].plot(np.arange(N_a), PAI, label='AI production')
        axs[0,0].legend()
        axs[0,0].set_ylabel('I and AI')
        axs[0,0].set_xlabel('Time in weeks')

        
        IIstar =   self.safe_storage['I']
        IAIPstar = self.safe_storage['AIP']
        axs[0,1].step(
            np.arange(N_s), SI, 
            where = 'post', c='darkblue', linestyle='-'
            ) 
        axs[0,1].plot([1, N_s], [IIstar, IIstar], c = 'darkblue', linestyle='--', label = 'I safe storage')
        axs[0,1].step(
            np.arange(N_s), SAIP, 
            where = 'post', c='darkorange', linestyle='-'
            )
        axs[0,1].plot([1, N_s], [IAIPstar, IAIPstar], c = 'darkorange', linestyle='--', label = 'AI primary safe storage')
        axs[0,1].legend()
        axs[0,1].set_xlabel('Time in weeks')

        axs[1,0].plot(np.arange(N_a), TP_Am, label='AI transport to America')
        axs[1,0].plot(np.arange(N_a), TP_As, label='AI transport to Asia')
        axs[1,0].legend()
        axs[1,0].set_ylabel('AI transport')
        axs[1,0].set_xlabel('Time in weeks')

        IAISstar_As = self.safe_storage['AIS_As']
        IAISstar_Am = self.safe_storage['AIS_Am']
        axs[1,1].step(
            np.arange(N_s), SAIS_As, 
            where = 'post', c='darkblue', linestyle='-'
            ) 
        axs[1,1].plot([1, N_s], [IAISstar_As, IAISstar_As], c='darkblue', linestyle='--', label = 'Asia AI safe storage')
        axs[1,1].step(
            np.arange(N_s), SAIS_Am, 
            where = 'post', c='darkorange', linestyle='-'
            ) 
        axs[1,1].plot([1, N_s], [IAISstar_Am, IAISstar_Am], c='darkorange', linestyle='--', label = 'America AI safe storage')
        axs[1,1].legend()
        axs[1,1].set_xlabel('Time in weeks')

        axs[3,0].plot(np.arange(N_a), P_PA, c = 'blue', label='PA production in Asia')
        axs[3,0].plot(np.arange(N_s), F_PA, c = 'cyan', linestyle = '--', label='PA forecast in Asia')
        axs[3,0].step(
            np.arange(N_a), np.array(self.demand_vector)[:,0], 
            where = 'post', c='darkblue', linestyle='-'
        ) 
        axs[3,0].legend()
        axs[3,0].set_ylabel('PA')
        axs[3,0].set_xlabel('Time in weeks')

        axs[2,0].step(
            np.arange(N_a), np.array(self.demand_vector)[:,2], 
            where = 'post', c='darkblue', linestyle='--'
            ) 
        axs[2,0].plot(np.arange(N_a), P_TEE, label='TEE production in Asia')
        axs[2,0].plot(np.arange(N_s), F_TEE, linestyle = '--', label='TEE forecast in Asia')
        axs[2,0].set_ylabel('TEE')
        axs[2,0].set_xlabel('Time in weeks')
        axs[2,0].legend()

        ITEEstar = self.safe_storage['TEE']

        axs[2,1].step(
            np.arange(N_s), S_TEE, 
            where = 'post', c='darkblue', linestyle='-'
        )     
        axs[2,1].plot([1, N_s], [ITEEstar, ITEEstar], linestyle = '--', label = 'TEE safe storage')
        axs[2,1].legend()
        axs[2,1].set_xlabel('Time in weeks')

        IPAstar = self.safe_storage['PA']
        axs[3,1].step(
            np.arange(N_s), S_PA, 
            where = 'post', c='darkorange', linestyle='-'
            ) 
        axs[3,1].plot([1, N_s], [IPAstar, IPAstar], linestyle = '--', c='darkorange', label = 'PA safe storage')
        axs[3,1].legend()
        axs[3,1].set_xlabel('Time in weeks')

        IPCstar = self.safe_storage['PC']
        axs[4,1].step(
            np.arange(N_s), S_PC, 
            where = 'post', c='darkblue', linestyle='-'
            ) 
        axs[4,1].plot([1, N_s], [IPCstar, IPCstar], linestyle = '--', c='darkblue', label = 'PC safe storage')
        axs[4,1].legend()
        axs[4,1].set_xlabel('Time in weeks')

        axs[4,0].plot(np.arange(N_a), P_PC, c='orange', label='PC production in America')
        axs[4,0].plot(np.arange(N_s), F_PC, linestyle = '--', c='green', label='PC forecast in America')
        axs[4,0].step(
            np.arange(N_a), np.array(self.demand_vector)[:,1], 
            where = 'post', c='darkorange', linestyle='-'
            ) 
        axs[4,0].legend()
        axs[4,0].set_ylabel('PC')
        axs[4,0].set_xlabel('Time in weeks')


        plt.show()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

# Create log dir
log_dir = "/tmp/gym/" ## stored in dv516/tmp
os.makedirs(log_dir, exist_ok=True)

env = SC_base(config)
check_env(env)
env = Monitor(env, log_dir)

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000, callback=callback)

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=5000)

results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "Supply Chain")

plot_results(log_dir)

# model = PPO.load('/tmp/gym/best_model.zip', env)

obs = env.reset()
for i in range(200):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        env.render()
        if env.cumulative_penalty*env.penalty_cost < 1e6:
            env.plots()
            obs = env.reset()
            break
        obs = env.reset()


# print('Random sampling: ')
# obs = env.reset()
# for i in range(100):
#     action = env.action_space.sample()
#     obs, rewards, done, info = env.step(action)
#     if done:
#         env.render()
#         env.plots()
#         obs = env.reset()




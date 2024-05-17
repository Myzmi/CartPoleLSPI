import numpy as np

from mushroom_rl.algorithms.value import LSPI
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.features import Features
from mushroom_rl.features.basis import PolynomialBasis, GaussianRBF
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter

import custom_cartpole

import tkinter as tk

# Create a window
root = tk.Tk()
root.title("Color Selector")
root.geometry('800x600')

# Configure column to expand horizontally
root.columnconfigure(0, weight=1)

massPole=2. #m
massCart=8. #M
poleLength=.5 #l

#add form frame
cart_frame = tk.LabelFrame(root, padx=20, pady=20, bd=0)
cart_frame.grid(row=0, column=0)

# Configure column to expand horizontally
cart_frame.columnconfigure(0, weight=1)

def experiment():
    np.random.seed()

    # MDP
    mdp = custom_cartpole.CustomCartPole(m=massPole, M=massCart, l=poleLength)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    basis = [PolynomialBasis()]

    s1 = np.array([-np.pi, 0, np.pi]) * .25
    s2 = np.array([-1, 0, 1])
    for i in s1:
        for j in s2:
            basis.append(GaussianRBF(np.array([i, j]), np.array([1.])))
    features = Features(basis_list=basis)

    fit_params = dict()
    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n,
                               phi=features)
    agent = LSPI(mdp.info, pi, approximator_params=approximator_params, fit_params=fit_params)


    # Algorithm
    core = Core(agent, mdp)
    core.evaluate(n_episodes=3, render=True)

    # Train
    core.learn(n_episodes=500, n_episodes_per_fit=500)

    # Test
    test_epsilon = Parameter(0.)
    agent.policy.set_epsilon(test_epsilon)

    dataset = core.evaluate(n_episodes=1, quiet=True)

    core.evaluate(n_steps=100, render=True)

    return np.mean(dataset.episodes_length)

def startClick():
    #get values
    global massCart
    global massPole
    global poleLength

    massCart= float(inputFieldCart.get())
    massPole= float(inputFieldPoleM.get())
    poleLength= float(inputFieldPoleL.get())

    root.destroy()

    mainE()

def mainE():
    n_experiment = 1

    logger = Logger(LSPI.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + LSPI.__name__)

    steps = experiment()
    logger.info('Final episode length: %d' % steps)

#cart_frame labels
formLabelCart = tk.Label(cart_frame, text="Cart Mass")
formLabelCart.grid(row=0, column=0, sticky="e")

formLabelPoleM = tk.Label(cart_frame, text="Pole Mass")
formLabelPoleM.grid(row=1, column=0, sticky="e")

formLabelPoleL = tk.Label(cart_frame, text="Pole Lenght")
formLabelPoleL.grid(row=2, column=0, sticky="e")

#cart_frame boxes
inputFieldCart = tk.Entry(cart_frame, width=10)
inputFieldCart.insert(0, "8.0")
inputFieldCart.grid(row=0, column=1, sticky="ew")

inputFieldPoleM = tk.Entry(cart_frame, width=10)
inputFieldPoleM.insert(0, "2.0")
inputFieldPoleM.grid(row=1, column=1, sticky="ew")

inputFieldPoleL = tk.Entry(cart_frame, width=10)
inputFieldPoleL.insert(0, "0.5")
inputFieldPoleL.grid(row=2, column=1, sticky="ew")

#start button
startButton = tk.Button(root, text="Start", command=startClick, width=25, height=3)
startButton.grid(row=2, column=0)

# Start the Tkinter event loop
root.mainloop()
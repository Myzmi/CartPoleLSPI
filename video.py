import gymnasium as gym
import custom_cartpole

import tkinter as tk

# Create a window
root = tk.Tk()
root.title("Color Selector")
root.geometry('800x600')

# Configure column to expand horizontally
root.columnconfigure(0, weight=1)

#add form frame
color_frame = tk.LabelFrame(root, padx=20, pady=20, bd=0)
color_frame.grid(row=0, column=0)

# Configure column to expand horizontally
color_frame.columnconfigure(0, weight=1)

#add form frame
cart_frame = tk.LabelFrame(root, padx=20, pady=20, bd=0)
cart_frame.grid(row=1, column=0)

# Configure column to expand horizontally
cart_frame.columnconfigure(0, weight=1)

colors={"backGr": (0, 176, 203),
		"cartColor": (22, 112, 19),
        "poleColor": (167, 255, 164)}

#functions
def setClick():
    #get color codes
    b1 = int(inputFieldB1.get())
    b2 = int(inputFieldB2.get())
    b3 = int(inputFieldB3.get())
    
    c1 = int(inputFieldC1.get())
    c2 = int(inputFieldC2.get())
    c3 = int(inputFieldC3.get())
    
    p1 = int(inputFieldP1.get())
    p2 = int(inputFieldP2.get())
    p3 = int(inputFieldP3.get())
    
	#set color codes
    colors["backGr"] = (b1, b2, b3)
    colors["cartColor"] = (c1, c2, c3)
    colors["poleColor"] = (p1, p2, p3)

    create_colored_circles()

def create_colored_circles():
    # Draw a circle on the canvas with the specified color
    canvas1.create_oval(5, 5, 30, 30, fill="#%02x%02x%02x" % colors["backGr"])
    canvas2.create_oval(5, 5, 30, 30, fill="#%02x%02x%02x" % colors["cartColor"])
    canvas3.create_oval(5, 5, 30, 30, fill="#%02x%02x%02x" % colors["poleColor"])

def startClick():
    #get values
    cartMass= float(inputFieldCart.get())
    poleMass= float(inputFieldPoleM.get())
    poleLenght= float(inputFieldPoleL.get())

    root.destroy()

    env = gym.make('CustomCartPole-v0', render_mode="human", selected_colors=colors, mass_cart= cartMass, mass_pole= poleMass, pole_lenght= poleLenght)
    observation, info = env.reset(seed=42)
    
    max_ep = 10
    
    for ep_cnt in range(max_ep):
        step_cnt = 0
        ep_reward = 0
        done = False
        
        state = env.reset()
        
        while not done:
            next_state, reward, done, _,_ = env.step(env.action_space.sample())
            env.render()
            step_cnt += 1
            ep_reward += reward
            state = next_state
            
        print('Episode: {}, Step count: {}, Episode reward: {}'.format(ep_cnt, step_cnt, ep_reward))
        
    env.close()


#labels
formLabelR = tk.Label(color_frame, text="R")
formLabelR.grid(row=0, column=1, sticky="ew")

formLabelG = tk.Label(color_frame, text="G")
formLabelG.grid(row=0, column=2, sticky="ew")

formLabelB = tk.Label(color_frame, text="B")
formLabelB.grid(row=0, column=3, sticky="ew")

formLabelBackGr = tk.Label(color_frame, text="Background Color")
formLabelBackGr.grid(row=1, column=0, sticky="e")

formLabelCart = tk.Label(color_frame, text="Cart Color")
formLabelCart.grid(row=2, column=0, sticky="e")

formLabelPole = tk.Label(color_frame, text="Pole Color")
formLabelPole.grid(row=3, column=0, sticky="e")

#boxes
#backgr
inputFieldB1 = tk.Entry(color_frame, width=5)
inputFieldB1.insert(0, "0")
inputFieldB1.grid(row=1, column=1, sticky="ew")

inputFieldB2 = tk.Entry(color_frame, width=5)
inputFieldB2.insert(0, "176")
inputFieldB2.grid(row=1, column=2, sticky="ew")

inputFieldB3 = tk.Entry(color_frame, width=5)
inputFieldB3.insert(0, "203")
inputFieldB3.grid(row=1, column=3, sticky="ew")

#cart
inputFieldC1 = tk.Entry(color_frame, width=5)
inputFieldC1.insert(0, "22")
inputFieldC1.grid(row=2, column=1, sticky="ew")

inputFieldC2 = tk.Entry(color_frame, width=5)
inputFieldC2.insert(0, "112")
inputFieldC2.grid(row=2, column=2, sticky="ew")

inputFieldC3 = tk.Entry(color_frame, width=5)
inputFieldC3.insert(0, "19")
inputFieldC3.grid(row=2, column=3, sticky="ew")

#pole
inputFieldP1 = tk.Entry(color_frame, width=5)
inputFieldP1.insert(0, "167")
inputFieldP1.grid(row=3, column=1, sticky="ew")

inputFieldP2 = tk.Entry(color_frame, width=5)
inputFieldP2.insert(0, "255")
inputFieldP2.grid(row=3, column=2, sticky="ew")

inputFieldP3 = tk.Entry(color_frame, width=5)
inputFieldP3.insert(0, "164")
inputFieldP3.grid(row=3, column=3, sticky="ew")

#test button
testButton = tk.Button(color_frame, text="Set Colors", command=setClick, width=15)
testButton.grid(row=4, column=0, columnspan=5)

#oval
canvas1 = tk.Canvas(color_frame, width=30, height=30)
canvas1.grid(row=1, column=4)

canvas2 = tk.Canvas(color_frame, width=30, height=30)
canvas2.grid(row=2, column=4)

canvas3 = tk.Canvas(color_frame, width=30, height=30)
canvas3.grid(row=3, column=4)

# Call function to draw colored circle on the canvas
create_colored_circles()

#cart_frame labels
formLabelCart = tk.Label(cart_frame, text="Cart Mass")
formLabelCart.grid(row=0, column=0, sticky="e")

formLabelPoleM = tk.Label(cart_frame, text="Pole Mass")
formLabelPoleM.grid(row=1, column=0, sticky="e")

formLabelPoleL = tk.Label(cart_frame, text="Pole Lenght")
formLabelPoleL.grid(row=2, column=0, sticky="e")

#cart_frame boxes
inputFieldCart = tk.Entry(cart_frame, width=10)
inputFieldCart.insert(0, "1.0")
inputFieldCart.grid(row=0, column=1, sticky="ew")

inputFieldPoleM = tk.Entry(cart_frame, width=10)
inputFieldPoleM.insert(0, "0.1")
inputFieldPoleM.grid(row=1, column=1, sticky="ew")

inputFieldPoleL = tk.Entry(cart_frame, width=10)
inputFieldPoleL.insert(0, "0.5")
inputFieldPoleL.grid(row=2, column=1, sticky="ew")

#start button
startButton = tk.Button(root, text="Start", command=startClick, width=25, height=3)
startButton.grid(row=2, column=0)

# Start the Tkinter event loop
root.mainloop()
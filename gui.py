import numpy as np
import tkinter as tk
import time
import threading

terrain_scanned = False

objects_scanned = False

running = True

def run_loop():
    global running
    while running:
        # Simulate some processing
        print("Running simulation...")
        # Simulate a delay
        time.sleep(1)
    return

def calibrate():
    print('calibrate/cancel')
    btn_A.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=50, pady=5)
    btn_L.grid(row=1, column=0, sticky="nsew", padx=50, pady=5)
    btn_R.grid(row=1, column=2, sticky="nsew", padx=50, pady=5)
    btn_Cancel.grid_remove()
    btn_Burn.grid_remove()
    btn_Scratch.grid_remove()
    btn_Water.grid_remove()
    return

def scan_terrain():
    print('scan_terrain')
    global terrain_scanned
    global objects_scanned
    terrain_scanned = True
    if objects_scanned:
        btn_B.config(state='normal')
    return

def scan_objects():
    print('scan_objects')
    global terrain_scanned
    global objects_scanned
    objects_scanned = True
    if terrain_scanned:
        btn_B.config(state='normal')
    return

def start_sim():
    print('start_sim')
    btn_B.grid_remove()
    btn_BL.grid(row=2, column=0, sticky="nsew", padx=50, pady=50)
    btn_BC.grid(row=2, column=1, sticky="nsew", padx=50, pady=50)
    btn_BR.grid(row=2, column=2, sticky="nsew", padx=50, pady=50)
    btn_BL.config(background='#F00')
    btn_BR['fg'] = '#00F'
    btn_A.config(state='disabled')
    btn_R.config(state='disabled')
    btn_L.config(state='disabled')
    threading.Thread(target=run_loop, daemon=True).start()
    return

def add_mitigation():
    print('add_mitigation')
    btn_A.grid_remove()
    btn_L.grid_remove()
    btn_R.grid_remove()
    btn_Cancel.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=50, pady=5)
    btn_Burn.grid(row=1, column=0, sticky="nsew", padx=50, pady=5)
    btn_Scratch.grid(row=1, column=1, sticky="nsew", padx=50, pady=5)
    btn_Water.grid(row=1, column=2, sticky="nsew", padx=50, pady=5)
    return


def end_sim():
    print('end_sim')
    global running
    running = False
    btn_BL.grid_remove()
    btn_BC.grid_remove()
    btn_BR.grid_remove()
    btn_B.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=50, pady=50)
    btn_A.config(state='normal')
    btn_R.config(state='normal')
    btn_L.config(state='normal')
    
    return

def pause_sim():
    global running
    if running:
        running = False
    else:
        running = True
        threading.Thread(target=run_loop, daemon=True).start()
    print('play/paused')

    return

gui = tk.Tk()
gui.title("AR Sandbox GUI")
gui.geometry("1200x800")

gui.grid_rowconfigure(0, weight=1)  
gui.grid_rowconfigure(1, weight=3) 
gui.grid_rowconfigure(2, weight=5)  

gui.grid_columnconfigure(0, weight=5)
gui.grid_columnconfigure(1, weight=5)
gui.grid_columnconfigure(2, weight=5)

# Top Section Calibrations
btn_A = tk.Button(gui, text="Calibrate", font=("Courier New", 20), command=calibrate)
btn_L = tk.Button(gui, text="Scan Terrain", font=("Courier New", 20), command=scan_terrain)
btn_R = tk.Button(gui, text="Scan Objects", font=("Courier New", 20), command=scan_objects)

# Top Section Mitigations
btn_Cancel = tk.Button(gui, text="Cancel Mitigation", font=("Courier New", 20), command=calibrate)
btn_Water = tk.Button(gui, text="Water Line", font=("Courier New", 20), command=scan_objects)
btn_Burn = tk.Button(gui, text="Burn Line", font=("Courier New", 20), command=scan_objects)
btn_Scratch = tk.Button(gui, text="Scratch Line", font=("Courier New", 20), command=scan_objects)

# Lower Section Start
btn_B = tk.Button(gui, text="Start Simulation", font=("Courier New", 20), command=start_sim)
btn_B.config(state='disabled')
# Lower Section Controls
btn_BL = tk.Button(gui, text="End Simulation", font=("Courier New", 20), command=end_sim)
btn_BC = tk.Button(gui, text="Pause Simulation", font=("Courier New", 20), command=pause_sim)
btn_BR = tk.Button(gui, text="Add Mitigation", font=("Courier New", 20), command=add_mitigation)


btn_A.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=50, pady=5)
btn_L.grid(row=1, column=0, sticky="nsew", padx=50, pady=5)
btn_R.grid(row=1, column=2, sticky="nsew", padx=50, pady=5)
btn_B.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=50, pady=50)

gui.mainloop()

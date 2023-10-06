import numpy as np
import matplotlib.pyplot as plt
# Constants
G = 6.674e-11 # Gravitational constant
M_E = 5.97e24  # Mass of Earth
M_M = 7.34e22 # Mass of Moon
R_E = 6.3781e6 # Radius of Earth
R_M = 1.7371e6 # Radius of Moon

# Initial conditions

m = 100000 # Mass of the rocket

def acc_x(x, y): # Acceleration of rocket in x direction function
    r = np.array([x, y]) # Position of rocket
    mag_r = np.sqrt(np.sum(r**2)) # Magnitude of position with respect to (0,0)
    r_M = np.array([0, 3.84e8]) # Position of Moon with respect to Earth (x,y)
    mag_r_M = np.sqrt(np.sum((r-r_M)**2))
    a_x = ((- G * M_E / mag_r**3) * x) - (G * M_M / mag_r_M**3) * (x - r_M[0])
    return a_x

def acc_y(x, y):  # Acceleration of rocket in y direction function
    r = np.array([x, y]) # Position of rocket
    mag_r = np.sqrt(np.sum(r**2)) # Magnitude of position with respect to (0,0)
    r_M = np.array([0, 3.84e8]) # Position of Moon with respect to Earth
    mag_r_M = np.sqrt(np.sum((r-r_M)**2))
    a_y = ((-G * M_E / mag_r**3) * y) - ((G * M_M / mag_r_M**3) * (y - r_M[1]))
    return a_y

def vel_x(vx): # Velocity in x direction function
    return vx

def vel_y(vy): # Velocity in y direction function
    return vy

def total_energy(x, y, vx, vy): # energy funciton
    ke = 0.5 * m * (vx**2 + vy**2)  # kinetic energy
    pe = -((G * M_E * m )/ np.sqrt(x**2 + y**2) ) - ((G * M_M * m) / (np.sqrt((x**2) + (y - 3.84e8)**2))) # Gravitational potential function 
    return ke + pe

def rk4(x, y, vx, vy): # Runge-Kutta 4th order function
    
    k1x = vel_x(vx)
    k1y = vel_y(vy)
    k1vx = acc_x(x, y)
    k1vy = acc_y(x, y)
    
    k2x = vel_x(vx + (dt*k1vx*0.5))
    k2y = vel_y(vy + (dt*k1vy*0.5))
    k2vx = acc_x(x + (0.5*dt*k1vx), y + (0.5*dt*k1vy))
    k2vy = acc_y(x + (0.5*dt*k1vx), y + (0.5*dt*k1vy))
    
    k3x = vel_x(vx + (dt*k2vx*0.5))
    k3y = vel_y(vy + (dt*k2vy*0.5))
    k3vx = acc_x(x + (0.5*dt*k2vx), y + (0.5*dt*k2vy))
    k3vy = acc_y(x + (0.5*dt*k2vx), y + (0.5*dt*k2vy))
    
    k4x = vel_x(vx + (dt*k3vx))
    k4y = vel_y(vy + (dt*k3vy))
    k4vx = acc_x(x + (dt*k3vx), y + (dt*k3vy))
    k4vy = acc_y(x + (dt*k3vx), y + (dt*k3vy))
    
    vx_new = vx + (1/6) * (k1vx + 2*k2vx + 2*k3vx + k4vx)
    vy_new = vy + (1/6) * (k1vy + 2*k2vy + 2*k3vy + k4vy)
    x_new = x + (1/6) * (k1x + 2*k2x + 2*k3x + k4x)
    y_new = y + (1/6) * (k1y + 2*k2y + 2*k3y + k4y)
    
    return vx_new, vy_new, x_new, y_new


MyInput = 0
while MyInput != 'q':
    print('Choice "a" is for a simulation of an orbit around earth')
    print('Choice "b" is for a simulation of a slingshot from the earth to the moon and back')
    print('')
    MyInput = input('Enter a choice, "a", "b" or "q" to quit: ' )
    print('You entered the choice: ',MyInput)
    if MyInput == 'a':
        print('You have chosen part (a): simulation of an orbit around earth') 
        dt = 0.2 # Time step in seconds
        t = np.arange(0, 6000, dt) # Time array

        pos = np.zeros((len(t), 2))
        vel = np.zeros((len(t), 2))
        M_M = 0
        x_0_rocket = R_E + 300000 # Initial x-position
        y_0_rocket = 0 # Initial y-position

        vx_0_rocket = 0 # Initial x-velocity
        vy_0_rocket = np.sqrt( G * M_E / x_0_rocket) # Initial y-velocity

        pos[0] = [x_0_rocket, y_0_rocket]
        vel[0] = [vx_0_rocket, vy_0_rocket]

        # Set up arrays to store the simulation results
        for i in range(1, len(t)):
            # Calculate new velocity in x and y directions
            vx_new, vy_new, x_new, y_new = rk4(pos[i-1, 0], pos[i-1, 1], vel[i-1, 0], vel[i-1, 1])
            #vy_new = vel_pos(pos[i-1, 0], pos[i-1, 1], vel[i-1, 0], vel[i-1, 1])

            # Update velocity array
            vel[i] = [vx_new, vy_new]

            # Update position array
            pos[i] = [x_new, y_new]
            # Check if the rocket has collided with the Earth
            if np.sqrt((x_new)**2 + (y_new)**2) <= R_E:
                print("Rocket has crashed into the Earth!")
                break
            
            # Check if the rocket has collided with the Moon
            elif np.sqrt((x_new-0)**2 + (y_new-3.84e8)**2) <= R_M:
                print("Rocket has landed on the Moon!")
            
            # Check if the rocket has returned to Earth
            elif i == len(t)-1:
                if np.sqrt((x_new)**2 + (y_new)**2) <= R_E:
                    print("Rocket has returned to Earth!")
                else:
                    print("Rocket is lost in space!")

        energy = np.zeros(len(t))
        energy[0] = total_energy(x_0_rocket, y_0_rocket, vx_0_rocket, vy_0_rocket)
        for i in range(1, len(t)):
            energy[i] = total_energy(pos[i, 0], pos[i, 1], vel[i, 0], vel[i, 1])
            
        # Create the orbital plot
        fig, ax = plt.subplots()
        center_circle = plt.Circle((0, 0), R_E, color='blue', fill=False)
        ax.add_artist(center_circle) # Adding the earths radius so an orbit can be seen
        ax.annotate('Earth radius', xy=(0, 0), xytext=(0,0), ha='center', va='center', fontsize=12)
        ax.plot(pos[:, 0], pos[:, 1])
        ax.axis('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Orbit of Rocket around Earth and Moon')

        # Create the energy plot
        fig, ax = plt.subplots()
        ax.plot(t, energy)
        ax.set_xlabel('Time (s)')
        ax.set_title('Energy Conservation of Rocket')
        ax.set_ylabel('Total energy (J)')

        # Show the plots
        plt.show()
        
    elif MyInput == 'b':
        print('You have chosen part (b): simulation of the slingshot to the moon') 
        dt = 0.2 # Time step in seconds
        t = np.arange(0, 644000, dt) # Time array
        pos = np.zeros((len(t), 2)) 
        vel = np.zeros((len(t), 2))
        r_0_rocket = 6.7e6 # Distance from Earth's center
        vx_0_rocket = 1.4019*np.sqrt( G * M_E / r_0_rocket) # Initial x-velocity
        vy_0_rocket = 0 # Initial y-velocity
        x_0_rocket = 0 # Initial x-position
        y_0_rocket = -r_0_rocket # Initial y-position

        pos[0] = [x_0_rocket, y_0_rocket]
        vel[0] = [vx_0_rocket, vy_0_rocket]

        # Set up arrays to store the simulation results
        for i in range(1, len(t)):
            # Calculate new velocity and position in x and y directions 
            vx_new, vy_new, x_new, y_new = rk4(pos[i-1, 0], pos[i-1, 1], vel[i-1, 0], vel[i-1, 1])
            vel[i] = [vx_new, vy_new]
            pos[i] = [x_new, y_new]
            
            # Check if the rocket has collided with the Earth
            if np.sqrt((x_new)**2 + (y_new)**2) <= R_E:
                print("Rocket has crashed into the Earth!")
                break
            
            # Check if the rocket has collided with the Moon
            elif np.sqrt((x_new-0)**2 + (y_new-3.84e8)**2) <= R_M:
                print("Rocket has landed on the Moon!")
            
            # Check if the rocket has returned to Earth
            elif i == len(t)-1:
                if np.sqrt((x_new)**2 + (y_new)**2) <= R_E:
                    print("Rocket has returned to Earth!")
                else:
                    print("Rocket is lost in space!")

        energy = np.zeros(len(t))
        energy[0] = total_energy(x_0_rocket, y_0_rocket, vx_0_rocket, vy_0_rocket)
        for i in range(1, len(t)):
            energy[i] = total_energy(pos[i, 0], pos[i, 1], vel[i, 0], vel[i, 1])
            
        plt.plot(pos[:, 0], pos[:, 1])
        plt.scatter([0], [0])
        plt.scatter([0], [3.84e8])
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Orbit of Rocket around Earth and moon')
        plt.show()
        plt.plot(t, energy)
        plt.xlabel('Time (s)')
        plt.title('Energy Conservation of Rocket')
        plt.ylabel('Total energy (J)')
        plt.show()
    elif MyInput != 'q':
        print ('This is not a valid choice')
        print('You have chosen to finish - goodbye.')

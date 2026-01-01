import pyvisa
import time
import os

FPS = 1.2 
time_interval = 1 / FPS  # Time interval in seconds
file_path = './data/bad_apple/commands.txt'

lines = []

with open(file_path, 'r') as file:
    for line in file:
        lines.append(line.strip())


if __name__ == "__main__":  

    # Initialize the Resource Manager
    rm = pyvisa.ResourceManager()

    # Open the HP8569B Spectrum Analyzer at the specified GPIB address (replace with correct GPIB address)
    try:
        hp8569b = rm.open_resource('GPIB0::1::INSTR')
        print("Successfully opened HP8569B")
    except Exception as e:
        print(f"Error opening HP8569B: {e}")
        exit()

    # Set timeout and termination characters for legacy instruments
    hp8569b.timeout = 5000
    hp8569b.read_termination = '\n'  # Use carriage return for legacy instruments
    hp8569b.write_termination = '\n'
    hp8569b.send_end = True  # Assert EOI at the end of write commands 
    
    # Start playing the video
    # The command below works on Windows to open the video with the default media player
    os.system('start bad_apple.mp4')

    time.sleep(0.7) # I had to add a delay to correctly sync the video and commands
    
    # Clear the text area of the spectrum analyzer
    hp8569b.write("LL ")
    hp8569b.write("LU ")
    
    # Send lines at the correct interval
    start_time = time.time()
    for i, line in enumerate(lines):
        print(f"Sending line {i} at {time.time()-start_time:.2f}s (Target: {(i)*time_interval:.2f}s)")
        hp8569b.write(line)
        next_time = start_time + ((i+1)*time_interval)
        sleep_time = next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
    


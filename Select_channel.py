import subprocess

STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
    'Pz', 'P4', 'T6', 'O1', 'O2'
]

for i in range(0, 17):
    for j in range(i+1, 18):
        for k in range(j+1, 19):
            SELECTED_CHANNELS = []
            SELECTED_CHANNELS.append(STANDARD_CHANNELS[i])
            SELECTED_CHANNELS.append(STANDARD_CHANNELS[j])
            SELECTED_CHANNELS.append(STANDARD_CHANNELS[k])

            subprocess.run(["python", "/home/tandat/Downloads/Projects/MF-MGCN/Fake_processing"] + SELECTED_CHANNELS) 
            subprocess.run(["python", "/home/tandat/Downloads/Projects/MF-MGCN/Fake_model.py"] + SELECTED_CHANNELS) 
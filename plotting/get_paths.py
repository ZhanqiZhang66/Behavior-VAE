# Created by zhanq at 4/22/2024
# File:
# Description:
# Scenario:
# Usage
import os
def get_my_path():
    myPath = dict()
    if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
        onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
        github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
        data_path = rf"C:\Users\zhanq\OneDrive - UC San Diego\SURF"
    elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
        github_path = r'D:\OneDrive - UC San Diego\GitHub'
        onedrive_path = r'D:\OneDrive - UC San Diego'
        data_path = rf"D:\OneDrive - UC San Diego\SURF"
    else:
        onedrive_path = r'C:\Users\kiet\OneDrive - UC San Diego'
        github_path = r'C:\Users\kiet\OneDrive - UC San Diego\GitHub'
        data_path = rf'C:\Users\kiet\OneDrive - UC San Diego\SURF'
    myPath['onedrive_path'] = onedrive_path
    myPath['github_path'] = github_path
    myPath['data_path'] = data_path
    return myPath




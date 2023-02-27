# Created by zhanq at 2/26/2023
# File:
# Description:
# Scenario:
# Usage
import os


if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    github_path = r'C:/Users/zhanq/OneDrive - UC San Diego/GitHub'  #
elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
    github_path = r'D:/OneDrive - UC San Diego/GitHub'  # Vic HOME
else:
    github_path = r'C:/Users/zhanq/OneDrive - UC San Diego/GitHub'  #

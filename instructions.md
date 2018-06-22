Alzheimer's Challenge Hackathon Instructions
====================


1. please add your name & email to this Google Sheet: https://tinyurl.com/y76vk384
2. in terminal, clone repo: $git clone https://github.com/swhustla/pycon2017-alzheimers-hack.git <destination folder>
3. cd to the <destination folder>
4. (optional) set up and enable a virtual env $ python3 -m venv my-project-env  
5. $pip3 install -r requirements.txt
6. $jupyter notebook
7. navigate to notebooks/tadpole_leaderboard.ipynb and get started

To make a simple submission:

run the leaderboard pipeline from notebooks/Makefile:

1. cd notebooks
2. make leaderboard
3. some errors will show up on purpose, as the name of the team needs to be setup. For fix them, set the right teamName in the Makefile and tadpole/forecast-simple.py
4. when th submission is generated, upload it to the TADPOLE website

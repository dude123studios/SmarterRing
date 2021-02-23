# SmarterRing
*Ring hacked with Deep Learning*

This is a repository that implements facial recognition on a model which I trained including facial detection with opencv, In order to recognize faces at your front door. It does this by implementing tensorflow, opencv, and a special unoficial ring api linked here- https://github.com/tchellomello/python-ring-doorbell. 

I am working to improve the detection system. It is trained on a 2GB Kaggle dataset of bollywood actors and actresses. I will continue expirimenting with Siamese model in order to generalize better. 

In order to use this with your own ring. use the following commands in a venv/ 

$git clone https://github.com/dude123studios/SmarterRing

$pip install -requirements.txt

Then setup 2 environment variables, USERNAME and PASSWORD, and make them equal to your ring acount's email and password

Finally, capture images all of your family members by running:

$python enter_image.py 

Then hitting space bar once there is a clear image of just one person, and enter their name. Once you see that all the images are in the data/faces/ directory, you are ready to start running the repository. Run:

$python main.py

It will ask you for a token in cmd, and the token should be sms'd to your phone. (This will only happen once, every time you run it after the first time this won't happen)

Once the doorbell is rung, you will see who is there. 


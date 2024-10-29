from thetadata import ThetaClient

# Launch the client to run in the background
client = ThetaClient(username="default", passwd="default", launch=True)

# This is where you run your code.

# Make sure to kill the client connection when you are done.
client.kill()
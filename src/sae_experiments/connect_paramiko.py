import os
import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
session = client.connect("beaker.cs.ucl.ac.uk", username="danietan", key_filename=os.path.expanduser("~/.ssh/id_ed25519"))
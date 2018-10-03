#!/usr/bin/env python

from socket import *
import os
import commands
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import time
import pickle
os.chdir("/home/sensorweb/Dropbox/DeCNN/")

MYPORT = 50000
Gamma = 1
n = 8
m = 1000
classNum = 18
hdunitNum = 4096
iteration = 50
## 64*18 chunks##
chunkSize = 64

## Function of check neighbor ##
def checkNeib():
    k = os.popen("ip route | awk -F\" \" '{ if($3 ==\"eth0\" && $1!=\"10.0.0.0/24\") print $1 }'|sort -R", "r")
    s = k.read()
    neibors = [s.strip() for s in s.splitlines()]
    return neibors
## 

## Function of getting ID ##
def getdit(myip):
    dit = int(str(myip.split()[0]).split(".")[3])
    return dit
##

## Getting neighbors and itself's ID##
time.sleep(20)
myip = commands.getoutput("hostname -I")
neis = checkNeib()
nei = [nb.split(".")[3] for nb in neis]
print(neis)
nodeId = getdit(myip)
print nodeId
##

s = socket(AF_INET, SOCK_DGRAM)
s.bind(('0.0.0.0', MYPORT))
s.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
s.settimeout(20.0)

testx = np.load("testx.npy")
testy = np.load("testy.npy")

for o in range(11):
    ## Getting the matrix##
    inibeta = np.load("beta"+str(o)+"_"+str(nodeId)+".npy")
    P = np.load("P"+str(o)+"_"+str(nodeId)+".npy")
    Pin = inv(P)
    ##

    matbeta={}

    for i in nei:
        matbeta[i] = np.zeros((iteration+1,inibeta.shape[0],inibeta.shape[1]))

    matbeta[str(nodeId)] = np.zeros((iteration+1,inibeta.shape[0],inibeta.shape[1]))
    matbeta[str(nodeId)][0] = inibeta

    for k in range(iteration):
        for j in range(hdunitNum/chunkSize):
            sendPack = [nodeId,j,matbeta[str(nodeId)][k,j*chunkSize:(j+1)*chunkSize,:],k]
            # print sendPack
            sendPackPick = pickle.dumps(sendPack)
            s.sendto(sendPackPick,('10.0.0.0', MYPORT))
            for i in range(len(nei)+1):
                try:
                    recvPackPick,addr = s.recvfrom(60000)
                    recvPack = pickle.loads(recvPackPick)
                    matbeta[str(recvPack[0])][recvPack[3],int(recvPack[1])*chunkSize:(int(recvPack[1])+1)*chunkSize,:] = recvPack[2]
                    recvK = recvPack[3]
                    # print "get "+str(recvPack[0])
                except timeout:
                    pass
            time.sleep(0.2)
        tempbeta = np.zeros((inibeta.shape[0],inibeta.shape[1]))
        for i in nei:
            tempbeta = np.add(tempbeta,matbeta[i][k]-matbeta[str(nodeId)][k]) 
        matbeta[str(nodeId)][k+1] = matbeta[str(nodeId)][k]+np.dot(Gamma*Pin,tempbeta)
        time.sleep(2)
        #print "Finished "+str(k)
        yhat2 = np.matmul(testx, matbeta[str(nodeId)][k+1])
        predict2 = np.argmax(yhat2, axis=1)
        test_accuracy  = np.mean(np.argmax(testy, axis=1) == predict2)
        print("iteration %d, phase %d, node %d, test accuracy = %.2f%%" % (k, o, nodeId, 100. * test_accuracy))
        #if k in [9,19,29,49,79,119]:
            #np.save("ringiter"+str(k)+"beta"+str(o)+"node"+str(nodeId)+".npy",matbeta[str(nodeId)][k+1])
            #print matbeta[str(nodeId)][k+1]
    






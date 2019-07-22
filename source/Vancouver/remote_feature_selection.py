#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:24:48 2019

@author: mc
"""


# =============================================================================
# feature selection
# =============================================================================
from paramiko import SSHClient
from scp import SCPClient
import os


def ssh_connection():
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('bigdatadb.polito.it',
                username='cocca',
                password=open('../../credential/bigdatadb.txt','r').readline(),
                look_for_keys=False)
    return ssh


def run_MRMR(ssh, train_norm, Label_name):
    col_to_remove = []
    for c in train_norm.columns:
        if Label_name == c: continue
        if ('sum' in c) or ('count' in  c)\
        or ('start' in c) or ('final'  in c)\
        or ('Gi_' in c) or ('t_age' in c):
            col_to_remove.append(c)
    
    
    train_norm.drop(col_to_remove, axis=1).to_csv('dataset_MRMR.csv')
    with SCPClient(ssh.get_transport()) as scp:
            scp.put('dataset_MRMR.csv', remote_path='/home/det_user/cocca/MicheleRankings/')
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('cd /home/det_user/cocca/MicheleRankings/; python Rankings.py %s'%Label_name)
    for line in ssh_stdout.readlines():
        print(line)
        
    for line in ssh_stderr.readlines():
        print(line)
    
    os.remove('dataset_MRMR.csv')
    
    
def download_all_outputs(ssh, Label_Name):
    with SCPClient(ssh.get_transport()) as scp:
        
        if not os.path.isdir('../../MicheleRankings/outputs/'):
            os.mkdir('../../MicheleRankings/outputs/')
        
        scp.get(remote_path='/home/det_user/cocca/MicheleRankings/outputs/%s'%Label_Name, 
                 local_path='../../MicheleRankings/outputs/', 
                 recursive=True)


def MRMR(train_norm, Label_name):
    ssh = ssh_connection()
    print('SSH connection established')
    run_MRMR(ssh,train_norm, Label_name)
    print('MRMR run')
    download_all_outputs(ssh, Label_name)
    print('Data downloaded')
    ssh.close()
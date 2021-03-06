# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2019 Dr. Helder Marchetto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Diamond: To connnect type:

Shortcut commands:
import sys
import os
sys.path.append(os.path.abspath("/dls_sw/i06/scripts/maccherozzi/TCP_Elmitec"))
import LeemConnect
oLeem = LeemConnect.oLeem()



1) Open a Terminal
2) Type in "python":
[i06user@i06-ws001 Desktop]$ python
3) Type in the following lines:
import sys
import os
sys.path.append(os.path.abspath("/dls_sw/i06/scripts/maccherozzi/TCP_Elmitec"))
import LeemConnect
4) Start the connection to leem2000 with:
oLeem = LeemConnect.oLeem()

The purpose of this Module is to create a wrapper for TCP commands for Leem2000
It can be used as follows:
oLeem = LeemConnect.oLeem(ip='localhost', port=5566, directConnect=True)
print(oLeem.getValue('FL'))

Use dir(oLeem) for a list of available methods
and help(oLeem) for an overview of the necessary parameters

The output of the help function will give a list of available methods. You do
not have to supply the "self" parameter.
This means that 

getHighLimit(self, module)

should be used as:

print(oLeem.getHighLimit('FL'))

"""

import socket
import time

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class oLeem(object):
    Leem2000Connected = False

    def __enter__(self):
        return self

    def __init__(self, ip= '172.23.106.79', port=5566, directConnect=True):
      if type(ip) <> str:
          print 'LEEM_Host must be a string. Using localhost instead.'
          self.ip = '172.23.106.79'
      else:
          self.ip = ip
      if type(port) <> int:
          print 'LEEM_Port must be an integer. Using 5566.'
          self.port = 5566
      else: 
          self.port = port
      self.lastTime=time.time()
      if directConnect:
          print 'Connect with ip='+str(self.ip)+', port='+str(self.port)
          self.connect()
          
    def __exit__(self, type, value, traceback):
        #print "Destroying", self
        if self.Leem2000Connected:
            #print 'Exit without closing connections... close connections'
            self.s.send('clo')
            self.s.close()
            self.Leem2000Connected = False
        

    def connect(self):
        if self.Leem2000Connected:
            print 'Already connected ... exit "connect" method'
            return
        else:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.ip, self.port))
            self.Leem2000Connected = True
            #Start string communication
            self.getTcp('asc', False, False, True)
            #Get list of devices
            self.updateModules()
            self.updateValues()
            print("Connected. %i modules found " % len(self.Mnemonic))
            print "#########################################"
            print "To see which devices are available, type:"
            print "for i in oLeem.Modules.values(): print i"
            print "for i in oLeem.Mnemonic.values(): print i"
            print "#########################################"
            print "To check the value of a device, type:"
            print "print oLeem.getValue('FL')"

    def testConnect(self):
        if self.Leem2000Connected:
            print 'Already connected ... exit "connect" method'
            return True
        else:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.settimeout(5) #5 seconds timeout
            t = time.time()
            try:
                self.s.connect((self.ip, self.port))
                print 'connected in '+str(time.time() - t)+' sec'
                self.s.settimeout(None)
                self.s.close()
                return True
            except:
                print 'connection not possible'
                print 'please check: that LEEM is running'
                print '              the IP address is correct'
                print '              the PORT is the same activated in LEEM2000'
                return False
            else:
                return False

    def setIP(self, IP):
        if self.Leem2000Connected:
            print 'Already connected ... close connection first'
            return
        else:
            if type(IP) != str:
                print 'The IP has to be a string. Please use this synthax:'
                print "object.setIP('192.168.1.0')"
                print "or"
                print "object.setIP('localhost')"
                return
            self.ip = str(IP)

    def setPort(self, port):
        if self.Leem2000Connected:
            print 'Already connected ... close connection first'
            return
        else:
            if type(port) != int:
                print 'The port has to be a number. Please use this synthax:'
                print "object.setPort(5566)"
                return
            self.port = str(port)

    def disconnect(self):
        if self.Leem2000Connected:
            self.s.send('clo')
            self.s.close()
            self.Leem2000Connected = False

    def updateValues(self):
        if not self.Leem2000Connected:
            print 'Please connect first'
            return None
        else:
            self.Values   = {}
            for x in self.Mnemonic:
                data = self.getTcp('get '+self.Modules[x], True, False, False)
                if is_number(data):
                    self.Values[x] = float(data)

    def updateModules(self):
        if not self.Leem2000Connected:
            print 'Please connect first'
            return None
        else:
            #Get list of devices
            self.nModules = self.getTcp('nrm', False, True, False)
            #Get list of devices
            self.Modules      = {}
            self.Mnemonic     = {}
            self.invModules   = {}
            self.invMnemonic  = {}
            self.lowLimit     = {}
            self.highLimit    = {}
            self.MnemonicUp   = {}
            self.ModulesUp    = {}
            for x in range(self.nModules):
                data = self.getTcp('nam '+str(x), False, False, True)
                if not data in ['', 'no name','invalid','disabled']:
                    self.Modules[x]   = data
                    self.ModulesUp[x] = data.upper()
                    self.invModules[data.upper()] = x
                data = self.getTcp('mne '+str(x), False, False, True)
                if not data in ['', 'no name','invalid','disabled']:
                    self.Mnemonic[x]   = data
                    self.MnemonicUp[x] = data.upper()
                    self.invMnemonic[data.upper()] = x
                ll = self.getLowLimit(x, isNotSetup=False)
                hl = self.getHighLimit(x, isNotSetup=False)
                if (not ll in ['', 'no name','invalid','disabled']) and is_number(ll):
                    self.lowLimit[x]  = ll
                if (not hl in ['', 'no name','invalid','disabled']) and is_number(hl):
                    self.highLimit[x] = hl

    def get(self, TCPString, module):
        #check if the input is a number or a string
        if TCPString[-1] <> ' ':
            TCPString+=' '
        if is_number(module):
            m = int(module)
            if self.Mnemonic.has_key(m):
                data = self.getTcp(TCPString+str(m), False, False, True)
                if not data in ['','invalid'] and is_number(data):
                    return float(data)
                else:
                    return 'invalid number '+str(m)+'. Return from leem='+str(data)
            else:
                return 'Module number '+str(module)+' not found'
        else:
            module = str(module)
            if module.upper() in self.invModules:
                data = self.getTcp(TCPString+(self.invModules[module.upper()]), False, False, True)
                if (not data in ['','invalid']) and is_number(data):
                    return float(data)
                else:
                    return 'invalid'
            elif module.upper() in self.invMnemonic:
                data = self.getTcp(TCPString+str(self.invMnemonic[module.upper()]), False, False, True)
                if not data in ['','invalid'] and is_number(data):
                    return float(data)
                else:
                    return 'invalid'
            else:
                return 'Module '+module+' not found'

    def getValue(self, module):
        if not self.Leem2000Connected:
            print 'Please connect first'
            return None
        else:
            #check if the input is a number or a string
            if (time.time()-self.lastTime) < 0.3:
                time.sleep(0.3)
            return self.get('get ',module)

    def setValue(self, module, value):
        if not self.Leem2000Connected:
            print 'Please connect first'
            return None
        else:
            #check if the input value is a number or a string
            if not is_number(value):
                return 'Value must be a number'
            else:
                value = str(value)
            #check if the input module is a number or a string
            self.lastTime=time.time()            
            if is_number(module):
                m = int(module)
                return self.getTcp('set '+str(m)+'='+value,False, False, True) == '0'
            else:
                if (module.upper() in self.MnemonicUp.values()) or (module.upper() in self.ModulesUp.values()):
                    return self.getTcp('set '+str(module)+'='+value,False, False, True) == '0'
                else:
                    return False

    def getLowLimit(self, module, isNotSetup=True):
        if not self.Leem2000Connected:
            print 'Please connect first'
            return None
        else:
            
            if (time.time()-self.lastTime) < 0.3 and isNotSetup:
                time.sleep(0.3)
            self.lastTime = time.time()
            TCPString = 'psl '
            return self.get(TCPString,module)

    def getHighLimit(self, module, isNotSetup=True):
        if not self.Leem2000Connected:
            print 'Please connect first'
            return None
        else:
            if (time.time()-self.lastTime) < 0.3 and isNotSetup:
                time.sleep(0.3)
            self.lastTime = time.time()
            TCPString = 'psh '
            return self.get(TCPString,module)

    def getFoV(self):
        if not self.Leem2000Connected:
            print 'Please connect first'
            return None
        else:
            #check if the input is a number or a string
            data=self.getTcp('prl',False, False, True)
            strSplit = data.partition(':')
            if strSplit[1] == ':':
                part = data.partition('\xb5')
                if part[1] == '\xb5':
                    FoVStr = part[0]
                    if is_number(FoVStr):
                        return (float(FoVStr), True) #True means that the FoV is a number
            return (data, False)   #False means that no useful number is given out

    def getModifiedModules(self):
        if not self.Leem2000Connected:
            print 'Please connect first'
            return None
        else:
            #check if the input is a number or a string
            data=self.getTcp('chm',False, False, True)
            if data != '0':
                arr = data.split()
                nChanges = int(arr[0])
                del arr[0]
                Changes = []
                for i in range(nChanges):
                    Changes.append({'moduleName':self.Modules[int(arr[i*2])], 'moduleNr':int(arr[i*2]), 'NewValue':float(arr[1+i*2])})
                return (nChanges,Changes)
            else:
                return (0,0)

    def getTcp(self, TCPString, isFlt=True, isInt=False, asIs=False):
        self.s.send(TCPString)
        retStr = self.TCPBlockingReceive()
        if asIs:
            #print 'is asIs = ', TCPString, retStr
            return retStr
        elif isInt:
            if is_number(retStr):
                return int(retStr)
            else:
                return 0
        elif isFlt:
            if is_number(retStr):
                return float(retStr)
            else:
                return 0.0
        else:
            return retStr

    def setTcp(self, TCPString, Value):
        TCPString = TCPString.strip()+' '+Value.strip()
        self.s.send(TCPString)
        return self.TCPBlockingReceive()

    def TCPBlockingReceive(self):
       Bytereceived = '0'
       szData = ''
       while ord(Bytereceived) != 0:
           ReceivedLength = 0
           while ReceivedLength == 0:
               Bytereceived = self.s.recv(1)
               #print 'Bytereceived=',Bytereceived,'ord(Bytereceived)=',ord(Bytereceived)
               ReceivedLength = len(Bytereceived)
           if ord(Bytereceived) != 0:
               szData = szData + Bytereceived
       return szData


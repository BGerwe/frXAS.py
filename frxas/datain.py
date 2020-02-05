import numpy as np
import pandas as pd
import glob 
import os
from matplotlib import pyplot as plt


def getdata(point, amplitude, base, initfile, finfile, makefig=False, XrayRaw=False):
	filestr=str(point + ' ' + str('%.3f'%float(amplitude)) + ' '+ "*.txt")
#Making array of strings to help pandas find all data files
# for a single measured amplitude and location (point)
	all_files=glob.glob(os.path.join(base,point, filestr))
	data=pd.concat((pd.read_csv(f,delimiter='\t') for f in all_files[initfile:finfile]),axis=1)
	nr=int(np.shape(data)[0]*np.shape(data)[1]/5) #desired total number of rows
	data.columns=np.tile(('Time','Io','If','J','V'),int(np.shape(data)[1]/5))
	
#Making time array
	dt=data.iloc[1,0]
	time=np.arange(0,nr)[:,]*dt
	time.resize(nr,1)

#Grab each signal into distinct arrays
	Io=np.array(data['Io'])
	If=np.array(data['If'])
	J=np.array(data['J'])
	V=np.array(data['V'])

#Reshape each array into single column arrays
	Io.resize(nr,1)
	If.resize(nr,1)
	J.resize(nr,1)
	V.resize(nr,1)
	
	if makefig==True:
		fig=plt.figure(figsize=(6.5,5))
		ax1 = fig.add_subplot(211)
		ax1.plot(time, V,time,J)

		ax2=fig.add_subplot(212)
		ax2.plot(time,Io,time,If)

		plt.show()
		
#Keeping option to report raw X-ray data, but it's usually uneccesary
	if XrayRaw==True:
		return time,Io, If, J, V
	else:
		Ir= If/Io
		return time,Ir, J, V

#function to get the applied frequency from data file header
def getfreq(point, amplitude, base):
    filestr=str(point + ' ' + str('%.3f'%float(1)) + ' '+ "*.txt")
    f=glob.glob(os.path.join(base,point, filestr))
    data=pd.read_csv(f[0],delimiter='\t')
    
    fa=np.array([float(data.columns[2])])
    return fa
	
def P2R(radii, angles):
    return radii * np.exp(1j*angles)

#function for getting the bin index of harmonic peaks in the fft
#Will return array of these indices depending on format of fft passed
#For "single" ffts (i.e. [-freq Re/Im -> +freq Re/Im]) will give
#[-first +first -second +second -third +third -fourth +fourth -fifth +fifth]
#For "combined ffts (i.e. [-freq Re +freq Re -freq Im +freq Im]) will give
#two columns of indices in form of "single" fft
def fftbin(freqin,freqlist, Ns,dt, FFTtype, harmonics):
#     tmeas=np.ceil(Ns*dt)
    tmeas=Ns*dt
        
    if FFTtype=="Real" or FFTtype=="Imag":
        mid=np.size(freqlist)/2
        bins=np.tile(mid,(2*harmonics))
        for i in range(0,harmonics):
            bins[2*i:2*i+2]=np.array(harm_switch(i+1,mid,tmeas,freqin))
        return bins
    elif FFTtype=="Combo":
        mid=np.size(freqlist)/4
        bins=np.tile(mid,(2*harmonics,2))
        for i in range(0,harmonics):
            bins[2*i:2*i+2,0]=np.array(harm_switch(i+1,mid,tmeas,freqin))
            bins[2*i:2*i+2,1]=np.array(harm_switch(i+1,mid*2,tmeas,freqin))
        return bins
    else:
        print('Invalid FFT type selection')
        return

#making switch case for finding harmonics fft bins

def first(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-freqin*tmeas)
    ind2=int(mid+freqin*tmeas)
    return ind1, ind2

def second(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-2*freqin*tmeas)
    ind2=int(mid+2*freqin*tmeas)
    return ind1, ind2

def third(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-3*freqin*tmeas)
    ind2=int(mid+3*freqin*tmeas)
    return ind1, ind2

def fourth(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-4*freqin*tmeas)
    ind2=int(mid+4*freqin*tmeas)
    return ind1, ind2

def fifth(mid,tmeas,freqin):#mid, freqin, tmeas):
    ind1=int(mid-5*freqin*tmeas)
    ind2=int(mid+5*freqin*tmeas)
    return ind1, ind2
    
switcher={
    1:first,
    2:second,
    3:third,
    4:fourth,
    5:fifth
}

#Function to evaluate switch
def harm_switch(arg,mid,tmeas,freqin):
    func=switcher.get(arg, "Wrong")
    
    if func=="Wrong":
        return "Invalid Harmonic"
    else:
        return func(mid,tmeas,freqin)

def getfft(data, Xrayraw):
    datafft=np.zeros(np.shape(data))+1j*0
    dt=data[0,1,0]
    Ns=np.shape(data)[1]
    freq=np.fft.fftshift(np.fft.fftfreq(Ns,dt))
    if Xrayraw==False:
        for n in range(0,np.shape(data)[2]):
            datafft[0,:,n]=np.fft.fftshift(np.fft.fft(data[0,:,n])/(np.sqrt(2)*Ns/2))
            datafft[1,:,n]=np.fft.fftshift(np.fft.fft(data[1,:,n])/(np.sqrt(2)*Ns/2))
            datafft[2,:,n]=np.fft.fftshift(np.fft.fft(data[2,:,n])/(np.sqrt(2)*Ns/2))
            datafft[3,:,n]=np.fft.fftshift(np.fft.fft(data[3,:,n])/(np.sqrt(2)*Ns/2))
        return freq, datafft
    else:
        for n in range(0,np.shape(data)[2]):
            datafft[0,:,n]=np.fft.fftshift(np.fft.fft(data[0,:,n])/(np.sqrt(2)*Ns/2))
            datafft[1,:,n]=np.fft.fftshift(np.fft.fft(data[1,:,n])/(np.sqrt(2)*Ns/2))
            datafft[2,:,n]=np.fft.fftshift(np.fft.fft(data[2,:,n])/(np.sqrt(2)*Ns/2))
            datafft[3,:,n]=np.fft.fftshift(np.fft.fft(data[3,:,n])/(np.sqrt(2)*Ns/2))
            datafft[4,:,n]=np.fft.fftshift(np.fft.fft(data[4,:,n])/(np.sqrt(2)*Ns/2))
        return freq, datafft
		
def Dawsonapp(b,fa,data):    
    dataD=np.array(data[0,:,:],ndmin=3)
    Dfunc=np.array(np.exp(-(fa*data[0,:,:]/b)**2),ndmin=3)
    dataD=data[1:,:,:]*Dfunc
    dataD=np.append(np.array(data[0,:,:],ndmin=3),dataD,axis=0)
    #print(np.shape(dataD),dataD[0,:,0],Dfunc)
    return dataD, Dfunc
	
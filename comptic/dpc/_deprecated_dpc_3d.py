"""
Implement 2D and 3D phase retrieval using differential phase contrast (DPC)

Michael Chen
Jun 2, 2017
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter
from labalg.iteralg import _norm,_softThreshold,_softThreshold_isoTV
from labalg.iteralg import gradientDescent,FISTA,newton,lbfgs
from labutil.displaytools import getROI
from labutil.opticstools import Fourier, pupilGen, _genGrid
pi    = np.pi
naxis = np.newaxis

class DPCSolver:
    def __init__(self, imgs, pixel_size, wavelength, NA, \
                 RI = 1, rotation = [0, 180, 90, 270], NA_in=0.0, DPC_num=4, cDPC=False):
        self.DPC_num      = 3 if cDPC else DPC_num
        self.intensity    = imgs.astype('float64')
        self.Fobj         = Fourier(imgs[0].shape,(-1,-2))
        self.normalization()
        self.NA           = NA
        self.NA_in        = NA_in
        self.pixel_size   = pixel_size
        self.wavelength   = wavelength
        self.RI           = RI
        self.pupil        = pupilGen(imgs[0].shape, pixel_size, NA, wavelength)
        self.fxlin        = np.fft.ifftshift(_genGrid(imgs[0].shape[1], 1.0/pixel_size/imgs[0].shape[1]))
        self.fylin        = np.fft.ifftshift(_genGrid(imgs[0].shape[0], 1.0/pixel_size/imgs[0].shape[0]))
        self.rotation     = rotation
        self.sourceGen()
        self.WOTFGen()
        self.reg_u        = 1e-6
        self.reg_p        = 1e-6
        self.reg_TV       = (1e-3,1e-3)
        self.rho          = 1e-3
        self.f,self.ax    = plt.subplots(1,3,figsize=(16,4))
        plt.close()

    def sourceGen(self):
        self.source = []
        P           = pupilGen(self.intensity[0].shape, self.pixel_size, self.NA, self.wavelength,\
                               NA_in = self.NA_in)
        for rotIdx in range(self.DPC_num):
            self.source.append(np.zeros(self.intensity[0].shape))
            rotdegree = self.rotation[rotIdx]
            if rotdegree <180:
                self.source[-1][self.fylin[:,naxis]*np.cos(np.deg2rad(rotdegree))+1e-15>=
                                self.fxlin[naxis,:]*np.sin(np.deg2rad(rotdegree))] = 1.0
                self.source[-1] *= P
            else:
                self.source[-1][self.fylin[:,naxis]*np.cos(np.deg2rad(rotdegree))+1e-15<
                                self.fxlin[naxis,:]*np.sin(np.deg2rad(rotdegree))] = -1.0
                self.source[-1] *= P
                self.source[-1] += P
        self.source = np.asarray(self.source)

    def WOTFGen(self):
        F  = lambda x: self.Fobj.fourierTransform(x)
        IF = lambda x: self.Fobj.inverseFourierTransform(x)
        self.Hu = []
        self.Hp = []
        for rotIdx in range(self.source.shape[0]):
            G        = ((self.RI/self.wavelength)**2-\
                        (self.fxlin[naxis,:]**2+self.fylin[:,naxis]**2))**0.5
            FSP_cFPG = F(self.source[rotIdx]*self.pupil)*F(self.pupil/G).conj()
            I0       = (self.source[rotIdx]*self.pupil*self.pupil.conj()).sum()
            self.Hu.append(2.0*IF(FSP_cFPG.real)*G/I0)
            self.Hp.append(2.0*1j*IF(1j*FSP_cFPG.imag)*G/I0)
        self.Hu = np.asarray(self.Hu)
        self.Hp = np.asarray(self.Hp)

    def normalization(self):
        self.ROI = getROI(self.intensity[0]/np.max(self.intensity[0]),'select a region without object')
        ROIsize  = (self.ROI[1]-self.ROI[0])*(self.ROI[3]-self.ROI[2])
        for img in self.intensity:
            img /= uniform_filter(img,size=img.shape[0]//2)
            meanIntensity = img[self.ROI[2]:self.ROI[3],self.ROI[0]:self.ROI[1]].sum()/ROIsize
            img /= meanIntensity        # normalize intensity with DC term
            img -= 1.0                  # subtract the DC term

    def plotResult(self,x,error):
        u_opt = x[:x.size//2].reshape(self.intensity[0].shape)
        p_opt = x[x.size//2:].reshape(self.intensity[0].shape)
        callback_amp_phase_error(self.f,self.ax,np.exp(u_opt+1j*p_opt),error,self.metadata.xlin.real,self.metadata.ylin.real)

    def afunc(self,x,forward_only=False,funcVal_only=False):
        F  = lambda x:self.Fobj.fourierTransform(x)
        IF = lambda x:self.Fobj.inverseFourierTransform(x)
        Fu = F(x[:x.size//2].reshape(self.intensity[0].shape))
        Fp = F(x[x.size//2:].reshape(self.intensity[0].shape))
        Ax = self.Hu*Fu[naxis,:,:] + self.Hp*Fp[naxis,:,:]
        Ax = np.asarray([IF(img_rotation).real for img_rotation in Ax])
        if forward_only:
            Ax.shape = (Ax.size,1)
            return Ax
        res     = Ax - self.intensity_current
        funcVal = _norm(res)**2
        if funcVal_only:
            return funcVal
        else:
            Fres              = np.asarray([F(img_rotation) for img_rotation in res])
            grad              = [IF((self.Hu.conj()*Fres).sum(axis=0)).real,IF((self.Hp.conj()*Fres).sum(axis=0)).real]
            grad              = np.append(grad[0].ravel(),grad[1].ravel())
            grad.shape        = (grad.size,1)
            grad[:x.size//2] += self.reg_u*x[:x.size//2]
            grad[x.size//2:] += self.reg_p*x[x.size//2:]
            return grad,funcVal

    def hessian(self,d):
        F    = lambda x:self.Fobj.fourierTransform(x)
        IF   = lambda x:self.Fobj.inverseFourierTransform(x)
        d_u  = d[:d.size//2].reshape(self.intensity[0].shape)
        d_p  = d[d.size//2:].reshape(self.intensity[0].shape)
        Fd_u = F(d_u); Fd_p = F(d_p);
        Hd = np.append((IF(((self.Hu.conj()*self.Hu).sum(axis=0))*Fd_u)+\
                       IF(((self.Hu.conj()*self.Hp).sum(axis=0))*Fd_p)+self.reg_u*d_u).real.ravel(),
                       (IF(((self.Hp.conj()*self.Hu).sum(axis=0))*Fd_u)+\
                       IF(((self.Hp.conj()*self.Hp).sum(axis=0))*Fd_p)+self.reg_p*d_p).real.ravel())
        Hd.shape = (Hd.size,1)
        return Hd

    def l2Deconv(self,fIntensity,AHA,determinant):
        IF  = lambda x: self.Fobj.inverseFourierTransform(x)
        AHy = [(self.Hu.conj()*fIntensity).sum(axis=0),(self.Hp.conj()*fIntensity).sum(axis=0)]
        u   = IF((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real
        p   = IF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real
        return u+1j*p

    def tvDeconv(self,fIntensity,AHA,determinant,fDx,fDy,order=1,TV_type='iso',maxIter=20):
        F       = lambda x: self.Fobj.fourierTransform(x)
        IF      = lambda x: self.Fobj.inverseFourierTransform(x)
        z_k     = np.zeros((4,)+self.intensity[0].shape)
        u_k     = np.zeros((4,)+self.intensity[0].shape)
        D_k     = np.zeros((4,)+self.intensity[0].shape)
        ROIsize = (self.ROI[1]-self.ROI[0])*(self.ROI[3]-self.ROI[2])
        for Iter in range(maxIter):
            y2  = [F(z_k[Idx] - u_k[Idx]) for Idx in range(self.DPC_num)]
            AHy = np.asarray([(self.Hu.conj()*fIntensity).sum(axis=0)+self.rho*(fDx.conj()*y2[0]+fDy.conj()*y2[1]),\
                              (self.Hp.conj()*fIntensity).sum(axis=0)+self.rho*(fDx.conj()*y2[2]+fDy.conj()*y2[3])])
            u   = IF((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real
            p   = IF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real
            if Iter < maxIter-1:
                if order==1:
                    D_k[0] = u - np.roll(u,-1,axis=1)
                    D_k[1] = u - np.roll(u,-1,axis=0)
                    D_k[2] = p - np.roll(p,-1,axis=1)
                    D_k[3] = p - np.roll(p,-1,axis=0)
                elif order==2:
                    D_k[0] = u - 2*np.roll(u,-1,axis=1) + np.roll(u,-2,axis=1)
                    D_k[1] = u - 2*np.roll(u,-1,axis=0) + np.roll(u,-2,axis=0)
                    D_k[2] = p - 2*np.roll(p,-1,axis=1) + np.roll(p,-2,axis=1)
                    D_k[3] = p - 2*np.roll(p,-1,axis=0) + np.roll(p,-2,axis=0)
                elif order==3:
                    D_k[0] = u - 3*np.roll(u,-1,axis=1) + 3*np.roll(u,-2,axis=1) - np.roll(u,-3,axis=1)
                    D_k[1] = u - 3*np.roll(u,-1,axis=0) + 3*np.roll(u,-2,axis=0) - np.roll(u,-3,axis=0)
                    D_k[2] = p - 3*np.roll(p,-1,axis=1) + 3*np.roll(p,-2,axis=1) - np.roll(p,-3,axis=1)
                    D_k[3] = p - 3*np.roll(p,-1,axis=0) + 3*np.roll(p,-2,axis=0) - np.roll(p,-3,axis=0)
                z_k = D_k + u_k
                if TV_type == 'iso':
                    z_k[:2,:,:] = _softThreshold_isoTV(z_k[:2,:,:],Lambda=self.reg_TV[0]/self.rho)
                    z_k[2:,:,:] = _softThreshold_isoTV(z_k[2:,:,:],Lambda=self.reg_TV[1]/self.rho)
                elif TV_type == 'aniso':
                    z_k[:2,:,:] = _softThreshold(z_k[:2,:,:],Lambda=self.reg_TV[0]/self.rho)
                    z_k[2:,:,:] = _softThreshold(z_k[2:,:,:],Lambda=self.reg_TV[1]/self.rho)
                else:
                    print('no such type for total variation!')
                    raise
                u_k += D_k - z_k
        return u+1j*p

    def solve(self,method='l2Deconv',xini=None,plot_verbose=False,**kwargs):
        self.method = method
        F     = lambda x: self.Fobj.fourierTransform(x)
        x_opt = []
        if plot_verbose:
            kwargs.update({'callback':self.plotResult})
        if self.method == 'l2Deconv':
            AHA = [(self.Hu.conj()*self.Hu).sum(axis=0)+self.reg_u,(self.Hu.conj()*self.Hp).sum(axis=0),\
                   (self.Hp.conj()*self.Hu).sum(axis=0),(self.Hp.conj()*self.Hp).sum(axis=0)+self.reg_p]
            if 'order' in kwargs:
                tv_order = kwargs['order']
                assert isinstance(tv_order,int), "order should be an integer!"
                assert tv_order > 0, "order should be possitive!"
                fDx      = np.zeros(self.intensity[0].shape)
                fDy      = np.zeros(self.intensity[0].shape)
                fDx[0,0] = 1.0; fDx[0,-1] = -1.0; fDx = F(fDx);
                fDy[0,0] = 1.0; fDy[-1,0] = -1.0; fDy = F(fDy);
                if tv_order > 1:
                    fDx = fDx**tv_order
                    fDy = fDy**tv_order
                reg_term= fDx*fDx.conj()+fDy*fDy.conj()
                AHA[0] += self.reg_TV[0]*reg_term
                AHA[3] += self.reg_TV[1]*reg_term
            determinant = AHA[0]*AHA[3]-AHA[1]*AHA[2]
            for frameIdx in range(self.intensity.shape[0]//self.DPC_num):
                fIntensity = np.asarray([F(self.intensity[frameIdx*self.DPC_num+imgIdx]) for imgIdx in range(self.DPC_num)])
                x_opt.append(self.l2Deconv(fIntensity,AHA,determinant))
            return x_opt
        elif self.method == 'tvDeconv':
            fDx      = np.zeros(self.intensity[0].shape)
            fDy      = np.zeros(self.intensity[0].shape)
            fDx[0,0] = 1.0; fDx[0,-1] = -1.0; fDx = F(fDx);
            fDy[0,0] = 1.0; fDy[-1,0] = -1.0; fDy = F(fDy);
            if 'order' not in kwargs or kwargs['order']==1:
                pass
            elif kwargs['order']==2:
                fDx = fDx**2; fDy = fDy**2;
            elif kwargs['order']==3:
                fDx = fDx**3; fDy = fDy**3;
            else:
                print('tvDeconv does not support order higher than 3!')
                raise

            reg_term     = self.rho*(fDx*fDx.conj()+fDy*fDy.conj())
            AHA         = [(self.Hu.conj()*self.Hu).sum(axis=0)+reg_term+self.reg_u,(self.Hu.conj()*self.Hp).sum(axis=0),\
                           (self.Hp.conj()*self.Hu).sum(axis=0),(self.Hp.conj()*self.Hp).sum(axis=0)+reg_term+self.reg_p]
            determinant = AHA[0]*AHA[3]-AHA[1]*AHA[2]
            for frameIdx in range(self.intensity.shape[0]//self.DPC_num):
                fIntensity = np.asarray([F(self.intensity[frameIdx*self.DPC_num+imgIdx]) for imgIdx in range(self.DPC_num)])
                x_opt.append(self.tvDeconv(fIntensity,AHA,determinant,fDx,fDy,**kwargs))
            return x_opt
        else:
            xini = np.zeros((2*self.intensity[0].size,1)) if xini is None else xini
            error = []
            if plot_verbose:
                kwargs.update({'callback':self.plotResult})
            for frameIdx in range(self.intensity.shape[0]//self.DPC_num):
                self.intensity_current = self.intensity[frameIdx*self.DPC_num:(frameIdx+1)*self.DPC_num]
                if self.method == 'gradientDescent':
                    x_opt_frame,error_frame = gradientDescent(self.afunc,xini,**kwargs)
                elif self.method == 'FISTA':
                    x_opt_frame,error_frame = FISTA(self.afunc,xini,**kwargs)
                elif self.method == 'newton':
                    x_opt_frame,error_frame = newton(self.afunc,xini,Hessian=self.hessian,**kwargs)
                else:
                    print('invalid method!')
                    raise
                x_opt.append(x_opt_frame[:x_opt_frame.size//2].reshape(self.intensity[0].shape)+\
                          1j*x_opt_frame[x_opt_frame.size//2:].reshape(self.intensity[0].shape))
                error.append(error_frame)
            return x_opt,error
